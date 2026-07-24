import torch.utils.data as data
import utils.utils_image as util


class BaseDataset(data.Dataset):
    """Shared base for KAIR/BasicSR-style datasets.

    Subclasses store their own configuration in ``__init__`` and implement
    ``_make_sample(self, img_H, index)`` -- the deterministic core transform
    that turns one loaded H (high-quality) image into the (H, L, aux) sample.
    ``__getitem__`` only needs to call ``self._load_img_H`` then
    ``self._make_sample`` and wrap the result into tensors.

    Keeping the core transform in ``_make_sample`` makes it directly
    unit-testable: a test can call ``dataset._make_sample(known_H, 0)`` with a
    known array instead of going through ``__getitem__`` (and disk I/O), which
    is exactly what ``tests/data/test_dataset_*.py`` does.

    If a subclass has no core transform of its own (it only loads a sample
    as-is), it may omit ``_make_sample`` and do the load + tensor wrapping
    entirely inside ``__getitem__``; the base default raises
    ``NotImplementedError``.

    Engineering discipline (this project):
      * ``n_channels`` is REQUIRED in ``opt`` -- no silent default.
      * ``paths_H`` is OPTIONAL. It must come from ``opt['paths_H']`` or
        ``util.get_image_paths(opt['dataroot_H'])``; if neither is given it is
        ``None``. This is intentional: synthesis/test-mode datasets
        (``DatasetDnCNN``/``FDnCNN``/``FFDNet``) build samples from a provided
        image via ``_make_sample`` and never touch disk, so they construct
        without paths. The guard is in ``__len__``: asking for the length of a
        paths-less dataset fails loud -- you never silently train on 0 images.
        Disk-backed subclasses (``DatasetDnPatch``/``Plain``/``PlainPatch``)
        additionally assert ``paths_H`` is present in their own ``__init__``.
      * ``paths_L`` is optional (``paths_L`` or ``dataroot_L``; else ``None``).
      * Invariants (positive ``n_channels``, ``paths_H`` is a list when given)
        are asserted at construction so misconfiguration fails loud, not at
        sample time 1000 images later.
      * ``opt`` itself is intentionally NOT exhaustively checked for unknown
        keys: this base is meant to be subclassed, and subclasses legitimately
        add their own keys. The keys this base consumes are strictly required.
    """

    def __init__(self, opt):
        super().__init__()
        # --- strict: opt must be a mapping ---
        assert isinstance(opt, dict), "opt must be a dict, got %r" % (type(opt),)

        # --- strict: n_channels is required, no silent default ---
        assert "n_channels" in opt, "opt must contain 'n_channels'"
        n_channels = opt["n_channels"]
        assert isinstance(n_channels, int), \
            "n_channels must be int, got %r" % (type(n_channels),)
        assert n_channels >= 1, \
            "n_channels must be >= 1, got %r" % (n_channels,)

        self.opt = opt
        self.n_channels = n_channels

        # --- paths_H is optional (synthesis mode may omit it) ---
        self.paths_H = self._resolve_image_paths(opt, "paths_H", "dataroot_H", required=False)
        assert self.paths_H is None or isinstance(self.paths_H, list), \
            "'paths_H' must resolve to a list or None, got %r" % (type(self.paths_H),)

        # --- paths_L is optional ---
        self.paths_L = self._resolve_image_paths(opt, "paths_L", "dataroot_L", required=False)

    @staticmethod
    def _resolve_image_paths(opt, paths_key, root_key, required):
        """Return the image path list for one side (H or L).

        Resolution rule (strict, no silent fallback):
          * if ``opt[paths_key]`` is present, use it directly (even if empty);
          * elif ``opt[root_key]`` is present, scan that directory;
          * else: raise if ``required``, else return ``None``.
        """
        has_paths = paths_key in opt
        has_root = root_key in opt
        if has_paths:
            return opt[paths_key]
        if has_root:
            return util.get_image_paths(opt[root_key])
        if required:
            raise AssertionError(
                "must provide '%s' or '%s' in opt (neither given)" % (paths_key, root_key)
            )
        return None

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

    def _make_sample(self, img_H, index):
        raise NotImplementedError
