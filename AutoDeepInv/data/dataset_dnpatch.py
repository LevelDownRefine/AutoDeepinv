import random
import numpy as np
import torch
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetDnPatch(BaseDataset):
    """Denoising dataset that pre-extracts H patches and corrupts them with fixed-sigma AWGN.

    All H patches are sampled once in ``__init__`` into ``self.H_data``. In train
    mode a patch is randomly flipped/rotated and corrupted with ``sigma/255``
    (PyTorch RNG) AWGN; in test mode the input image is degraded with seeded
    ``sigma_test/255`` numpy AWGN (deterministic, testable).
    """
    _requires_H = True  # disk-backed: loads H from disk via _load_img_H

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # DnPatch hyperparameters are REQUIRED from the sweep/config layer -- no
        # silent defaults, sigma_test included (see DatasetDnCNN for why the
        # train==test assumption is not a safe fallback).
        self.patch_size = self._pop_kwargs(self._kwargs, 'H_size')
        self.sigma = self._pop_kwargs(self._kwargs, 'sigma')
        self.sigma_test = self._pop_kwargs(self._kwargs, 'sigma_test')
        self.num_patches_per_image = self._pop_kwargs(self._kwargs, 'num_patches_per_image')
        self.num_sampled = self._pop_kwargs(self._kwargs, 'num_sampled')
        assert not self._kwargs, "unknown DatasetDnPatch keys: %s" % sorted(self._kwargs)
        del self._kwargs

        # Disk-backed dataset: paths_H is mandatory (synthesis mode is not valid
        # here, since patches are pre-extracted from disk in update_data()). The
        # requirement is enforced at resolution time via _requires_H = True, so a
        # missing/empty paths_H already failed loud in __init__ (super().__init__).

        self.num_sampled = min(self.num_sampled, len(self.paths_H))

        self.total_patches = self.num_sampled * self.num_patches_per_image
        self.H_data = np.zeros([self.total_patches, self.patch_size, self.patch_size, self.n_channels], dtype=np.uint8)
        self.update_data()

    def update_data(self):
        """Sample ``num_sampled`` images and extract ``num_patches_per_image`` random patches from each."""
        self.index_sampled = random.sample(range(0, len(self.paths_H), 1), self.num_sampled)
        n_count = 0
        for i in range(len(self.index_sampled)):
            H_patches = self.get_patches(self.index_sampled[i])
            for H_patch in H_patches:
                self.H_data[n_count, :, :, :] = H_patch
                n_count += 1

    def get_patches(self, index):
        """Read image ``index`` and crop ``num_patches_per_image`` random patches from it."""
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        H, W = img_H.shape[:2]
        H_patches = []
        num = self.num_patches_per_image
        for _ in range(num):
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            H_patches.append(img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :])
        return H_patches

    def _load_img_H(self, index):
        """Return the pre-extracted H patch (train) or read the full H image (test) from disk."""
        if self.phase == 'train':
            return self.H_data[index]
        return util.imread_uint(self.paths_H[index], self.n_channels)

    def __len__(self):
        return len(self.H_data)

    def _make_sample(self, img_H):
        """Build (H, L): train augments a patch then adds torch AWGN(sigma); test adds seeded numpy AWGN(sigma_test)."""
        if self.phase == 'train':
            mode = random.randint(0, 7)
            img_H = util.augment_img(img_H, mode=mode)
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            noise = torch.randn(img_L.shape).mul_(self.sigma / 255.0).numpy()
            img_L = img_L + noise
            return img_H, img_L
        else:
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test / 255.0, img_L.shape)
            return img_H, img_L

    def __getitem__(self, index):
        """Return ``{'img_L', 'img_H'}`` as float32 tensors."""
        img_H = self._load_img_H(index)
        img_H, img_L = self._make_sample(img_H)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        return {'img_L': img_L, 'img_H': img_H}
