import random
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetPlain(BaseDataset):
    """Image-to-image mapping dataset: loads both L and H (paths_L and paths_H required)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Image-to-image mapping: both L and H are provided on disk (no
        # synthesis). patch_size (H_size) is REQUIRED from the sweep/config
        # layer -- no silent 64 default.
        self.patch_size = self._pop_kwargs(self._kwargs, 'H_size')
        assert not self._kwargs, "unknown DatasetPlain keys: %s" % sorted(self._kwargs)
        del self._kwargs

        # Disk-backed dataset: both paths are mandatory and must be paired.
        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty. Plain dataset assumes both L and H are given!'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def _make_sample(self, img_H, img_L):
        """Build (H, L): test returns the full paired images; train crops+augments a paired patch."""
        if self.phase == 'train':
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the L/H patch pair
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)

            img_L, img_H = patch_L, patch_H

        return img_H, img_L

    def __getitem__(self, index):
        """Return ``{'img_L', 'img_H'}`` as float32 tensors (uint8->tensor)."""
        img_H = self._load_img_H(index)
        img_L = self._load_img_L(index)
        img_H, img_L = self._make_sample(img_H, img_L)
        img_H, img_L = util.uint2tensor3(img_H), util.uint2tensor3(img_L)
        return {'img_L': img_L, 'img_H': img_H}
