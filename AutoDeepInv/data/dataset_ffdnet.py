import random
import numpy as np
import torch
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetFFDNet(BaseDataset):
    """FFDNet dataset: L = H + AWGN(sigma); sigma (noise level) is returned separately as ``C``.

    sigma is drawn per sample from [sigma_min, sigma_max] (train) or fixed at
    sigma_test (test, seeded for reproducibility); the network is fed sigma as a
    conditioning input rather than via a concatenated noise-level map.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # FFDNet hyperparameters are REQUIRED from the sweep/config layer -- no
        # silent defaults. sigma is a [min,max] training *range* and sigma_test
        # is a separate scalar test level (cannot be derived from sigma), so it
        # must be supplied explicitly (no magic 25).
        self.patch_size = self._pop_kwargs(self._kwargs, 'H_size')
        self.sigma = self._pop_kwargs(self._kwargs, 'sigma')
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = self._pop_kwargs(self._kwargs, 'sigma_test')
        assert not self._kwargs, "unknown DatasetFFDNet keys: %s" % sorted(self._kwargs)
        del self._kwargs

    def _make_sample(self, img_H):
        """Build (H, L, noise_level): train crops+augments then adds AWGN(sigma); test adds seeded AWGN(sigma_test).

        Returns the per-sample noise level (sigma/255) as a scalar so ``__getitem__``
        can hand it to the network as the ``C`` tensor.
        """
        if self.phase == 'train':
            # get L/H/sigma patch pairs
            H, W, _ = img_H.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # augmentation - flip, rotate
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            img_H = util.uint2single(patch_H)
            img_L = np.copy(img_H)
            noise_level = np.random.uniform(self.sigma_min, self.sigma_max) / 255.0
            noise = (np.random.randn(*img_L.shape).astype(np.float32)) * np.float32(noise_level)
            img_L += noise
        else:
            # get L/H/sigma image pairs (deterministic -> directly testable)
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test / 255.0, img_L.shape)
            noise_level = self.sigma_test / 255.0

        return img_H, img_L, noise_level

    def __getitem__(self, index):
        """Return ``{'img_L', 'img_H', 'C'}`` as float32 tensors (C = noise level)."""
        img_H = self._load_img_H(index)
        img_H, img_L, noise_level = self._make_sample(img_H)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        C = torch.FloatTensor([noise_level]).view(1, 1, 1)
        return {'img_L': img_L, 'img_H': img_H, 'C': C}
