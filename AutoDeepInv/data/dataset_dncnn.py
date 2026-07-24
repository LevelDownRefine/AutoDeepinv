import random
import numpy as np
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetDnCNN(BaseDataset):
    """Denoising dataset for fixed-sigma AWGN (e.g. DnCNN): L = H + AWGN(sigma/255)."""

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__(opt)
        # DnCNN hyperparameters are REQUIRED from the sweep/config layer -- no
        # silent defaults. sigma_test is required too: the "test level == train
        # sigma" coincidence only holds when the sweep actually tests at the
        # train level; if it tests at a different level and forgets sigma_test,
        # a silent fallback would be exactly the misconfiguration we forbid.
        self.patch_size = self._demand(opt, 'H_size')
        self.sigma = self._demand(opt, 'sigma')
        self.sigma_test = self._demand(opt, 'sigma_test')

    def _make_noisy(self, img, sigma, seed=None):
        """Add zero-mean AWGN of std ``sigma/255`` using ``np.random.normal``.

        seed given  -> ``np.random.seed(seed)`` is set first, so the noise is
                       reproducible (test path, exact-assertable in value tests)
        seed None   -> global RNG used as-is -> fresh random noise each call
                       (train path), aligned with the original implementation.
        """
        if seed is not None:
            np.random.seed(seed)
        return img + np.random.normal(0, sigma / 255.0, img.shape)

    def _make_sample(self, img_H, index):
        """Build (H, L): train crops+augments then adds AWGN(sigma); test adds seeded AWGN(sigma_test)."""
        if self.opt['phase'] == 'train':
            # get L/H patch pairs
            H, W, _ = img_H.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # augmentation - flip, rotate
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            img_H = util.uint2single(patch_H)
            img_L = np.copy(img_H)
            img_L = self._make_noisy(img_L, self.sigma)
            return img_H, img_L
        else:
            # get L/H image pairs (deterministic -> directly testable)
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            img_L = self._make_noisy(img_L, self.sigma_test, seed=0)
            return img_H, img_L

    def __getitem__(self, index):
        """Return ``{'L', 'H', 'H_path', 'L_path'}`` as float32 tensors."""
        H_path = self.paths_H[index] if self.paths_H is not None else ''
        img_H = self._load_img_H(index)
        img_H, img_L = self._make_sample(img_H, index)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': H_path}
