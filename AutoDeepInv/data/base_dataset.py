import torch.utils.data as data
import utils.utils_image as util


class BaseDataset(data.Dataset):
    """Shared base for KAIR/BasicSR-style datasets.

    Subclasses store their own configuration in ``__init__`` and implement
    ``_make_sample(self, img_H)`` -- the deterministic core transform that turns
    one loaded H (high-quality) image into the (H, L, aux) sample. Paired /
    image-to-image datasets (e.g. ``DatasetPlain``) take the extra ``img_L``
    argument they need; the ``index`` is intentionally NOT threaded
    through -- it is used only inside ``__getitem__`` to load the on-disk H (and
    L) image, never inside the core transform. This keeps ``_make_sample`` a pure
    function of its image inputs, which is what makes it directly unit-testable.
    ``__getitem__`` only needs to call ``self._load_img_H`` then
    ``self._make_sample`` and wrap the result into tensors.

    Keeping the core transform in ``_make_sample`` makes it directly
    unit-testable: a test can call ``dataset._make_sample(known_H)`` with a
    known array instead of going through ``__getitem__`` (and disk I/O), which
    is exactly what ``tests/data/test_dataset_*.py`` does.

    If a subclass has no core transform of its own (it only loads a sample
    as-is), it may omit ``_make_sample`` and do the load + tensor wrapping
    entirely inside ``__getitem__``; the base default raises
    ``NotImplementedError``.

    Engineering discipline (this project):
      * ``n_channels`` is REQUIRED in ``kwargs`` -- no silent default.
      * ``paths_H`` / ``paths_L`` are resolved from ``kwargs['paths_H']`` /
        ``dataroot_H`` (and likewise for L). Whether a side is *required* is a
        per-subclass declaration (``_requires_H`` / ``_requires_L`` class
        attributes), NOT a separate post-hoc ``assert``. The base leaves both
        False so abstract/stub/synthesis datasets construct without on-disk
        images; a subclass that loads H via ``_load_img_H`` sets
        ``_requires_H = True`` so a missing/empty ``paths_H`` fails loud *at
        construction* (inside ``_resolve_image_paths`` with ``required=True``)
        -- never later in ``__getitem__`` / ``__len__``. Disk-backed datasets
        (``DatasetDnPatch``/``Plain``/``PlainPatch``) require H; ``Plain`` /
        ``PlainPatch`` additionally require L (paired image-to-image mapping).
      * Invariants (positive ``n_channels``, ``paths_H`` is a list when given)
        are asserted at construction so misconfiguration fails loud, not at
        sample time 1000 images later.
      * ``kwargs`` are STRICTLY checked (project discipline §4): the base
        consumes its own keys (``n_channels``, ``phase``, ``paths_H`` /
        ``dataroot_H``, ``paths_L`` / ``dataroot_L``) and hands the *remainder*
        to the subclass via ``self._kwargs``. After consuming its hyperparameters
        the subclass must ``assert not self._kwargs`` (any leftover key is a
        typo'd or schema-drifted parameter and fails loud) and then
        ``del self._kwargs`` so the half-consumed dict can't linger and be
        misread later. This is the "no silent unknown key" half of the kwargs
        discipline, applied to the data layer exactly as to the automation layer.
      * Subclasses (the denoising datasets) require their own hyperparameters
        (``H_size``, ``sigma``, ``sigma_test``, ``num_patches_per_image``,
        ``num_sampled``, ...) via ``BaseDataset._pop_kwargs(self._kwargs, key)`` --
        no silent ``.get()`` defaults, and every demanded key is *popped* so it
        cannot be mistaken for a leftover. The sweep/config layer is the single
        source of these values.
    """

    # Path requirements are declared per subclass, NOT checked by a separate
    # post-hoc assert. A subclass that loads H (resp. L) from disk via
    # ``_load_img_H`` (resp. ``_load_img_L``) sets the matching flag to True; the
    # base then enforces presence *at resolution time* (``_resolve_image_paths``
    # with ``required=True``), so a missing/empty path fails loud during
    # construction -- not later in ``__getitem__`` / ``__len__``. The base leaves
    # both False so abstract/stub datasets stay constructible without on-disk
    # images (synthesis mode).
    _requires_H = False
    _requires_L = False

    def __init__(self, **kwargs):
        super().__init__()

        # --- strict: n_channels is required, no silent default ---
        assert "n_channels" in kwargs, "kwargs must contain 'n_channels'"
        n_channels = kwargs.pop("n_channels")
        assert isinstance(n_channels, int), \
            "n_channels must be int, got %r" % (type(n_channels),)
        assert n_channels >= 1, \
            "n_channels must be >= 1, got %r" % (n_channels,)
        self.n_channels = n_channels

        # --- strict: phase is required and must be a known mode ---
        assert "phase" in kwargs, "kwargs must contain 'phase'"
        phase = kwargs.pop("phase")
        assert phase in ("train", "test", "val"), \
            "phase must be one of train/test/val, got %r" % (phase,)
        self.phase = phase

        # --- paths_H: required only if the subclass declares _requires_H
        #     (it loads H via _load_img_H). Enforced at resolution time below,
        #     so a missing/empty path fails loud here, not in __getitem__. ---
        self.paths_H = self._resolve_image_paths(
            kwargs, "paths_H", "dataroot_H", required=self._requires_H)
        assert self.paths_H is None or isinstance(self.paths_H, list), \
            "'paths_H' must resolve to a list or None, got %r" % (type(self.paths_H),)

        # --- paths_L: required only if the subclass declares _requires_L ---
        self.paths_L = self._resolve_image_paths(
            kwargs, "paths_L", "dataroot_L", required=self._requires_L)

        # The remaining kwargs are the subclass's hyperparameters, exposed as
        # ``self._kwargs`` for the subclass to consume. The subclass pops each
        # required key via ``self._pop_kwargs(self._kwargs, key)``, then asserts
        # ``not self._kwargs`` and ``del``s it so the half-consumed dict can't
        # linger and be misread later.
        self._kwargs = kwargs

    def _resolve_image_paths(self, kwargs, paths_key, root_key, required):
        """Resolve the image path list for one side (H or L), enforcing presence when ``required``.

        Resolution rule (strict, no silent fallback):
          * if ``kwargs[paths_key]`` is present, use it directly (even if empty)
            and consume that key;
          * elif ``kwargs[root_key]`` is present, scan that directory and consume
            that key;
          * else: ``paths`` is ``None``.

        When ``required`` is true the path MUST be provided and non-empty -- the
        check happens HERE, at resolution time (construction), so a missing or
        empty path fails loud immediately instead of later inside
        ``__getitem__`` / ``__len__``. Subclasses declare their needs via the
        ``_requires_H`` / ``_requires_L`` class attributes; the base leaves both
        False so abstract/stub datasets stay constructible without on-disk
        images (synthesis mode).
        The consumed key is popped from ``kwargs`` so it does not survive as a
        "leftover unknown key" when the subclass later asserts ``not kwargs``.
        """
        has_paths = paths_key in kwargs
        has_root = root_key in kwargs
        if has_paths:
            paths = kwargs.pop(paths_key)
        elif has_root:
            paths = util.get_image_paths(kwargs.pop(root_key))
        else:
            paths = None
        if required and not paths:
            raise AssertionError(
                "%s requires '%s' (or '%s') to be present and non-empty; got %r"
                % (type(self).__name__, paths_key, root_key, paths)
            )
        return paths

    def _pop_kwargs(self, kwargs, key):
        """Require ``key`` in ``kwargs`` and return it -- fail loud, never default.

        Centralizes the "no silent default" discipline for subclass
        hyperparameters. Instead of ``kwargs.get('H_size', 64)`` (which hides a
        misconfiguration behind a magic 64), a subclass does
        ``self.patch_size = self._pop_kwargs(self._kwargs, 'H_size')`` and gets a clear,
        class-tagged AssertionError when the sweep forgot to pass it. The owning
        class name is taken automatically from ``type(self).__name__``, so call
        sites never pass it (and cannot drift or typo it). The key is popped, so
        a successfully-demanded hyperparameter is never mistaken for a leftover
        unknown key.
        """
        owner = type(self).__name__
        assert key in kwargs, \
            "%s requires '%s' in kwargs -- no default value is allowed " \
            "(supply it from the sweep/config layer)" % (owner, key)
        return kwargs.pop(key)

    def _load_img_H(self, index):
        """Read a uint8 HWC (RGB) high-quality image for ``index`` from disk."""
        assert self.paths_H is not None, \
            "paths_H is None; cannot load image %r" % (index,)
        assert 0 <= index < len(self.paths_H), \
            "index %r out of range for paths_H (len %d)" % (index, len(self.paths_H))
        return util.imread_uint(self.paths_H[index], self.n_channels)

    def _load_img_L(self, index):
        """Read a uint8 HWC (RGB) low-quality image for ``index`` from disk."""
        assert self.paths_L is not None, \
            "paths_L is None; cannot load image %r" % (index,)
        assert 0 <= index < len(self.paths_L), \
            "index %r out of range for paths_L (len %d)" % (index, len(self.paths_L))
        return util.imread_uint(self.paths_L[index], self.n_channels)

    def __len__(self):
        # Fail loud when the dataset was built in synthesis mode (paths_H is
        # None). A silent 0 here would let a misconfigured training run proceed
        # on an empty dataset. Disk-backed datasets resolve a real path list.
        assert self.paths_H is not None, \
            "paths_H is None: this dataset has no on-disk images. " \
            "Either provide 'paths_H'/'dataroot_H', or use _make_sample(...) " \
            "directly with a provided image (synthesis/test mode)."
        return len(self.paths_H)

    def _make_sample(self, img_H):
        raise NotImplementedError
