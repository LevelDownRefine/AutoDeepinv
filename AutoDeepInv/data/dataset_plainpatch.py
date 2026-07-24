import random
import numpy as np
import utils.utils_image as util
from data.base_dataset import BaseDataset


class DatasetPlainPatch(BaseDataset):
    """Image-to-image mapping dataset that pre-extracts L/H patch pairs into buffers; both paths_L and paths_H required."""

    def __init__(self, opt):
        super(DatasetPlainPatch, self).__init__(opt)
        # Image-to-image mapping with pre-extracted L/H patch buffers. 64x64
        # default crop; 40 patches/image, 3000 sampled images by default.
        self.patch_size = self.opt.get('H_size', 64)

        self.num_patches_per_image = opt.get('num_patches_per_image', 40)
        self.num_sampled = opt.get('num_sampled', 3000)

        # Disk-backed dataset: both paths mandatory and must be paired.
        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty. This dataset uses L path, you can use dataset_dnpatch'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'H and L datasets have different number of images - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

        # ------------------------------------
        # number of sampled images
        # ------------------------------------
        self.num_sampled = min(self.num_sampled, len(self.paths_H))

        # ------------------------------------
        # reserve space with zeros
        # ------------------------------------
        self.total_patches = self.num_sampled * self.num_patches_per_image
        self.H_data = np.zeros([self.total_patches, self.patch_size, self.patch_size, self.n_channels], dtype=np.uint8)
        self.L_data = np.zeros([self.total_patches, self.patch_size, self.patch_size, self.n_channels], dtype=np.uint8)

        # ------------------------------------
        # update H/L patches
        # ------------------------------------
        self.update_data()

    def update_data(self):
        """Sample images and fill the H/L patch buffers from disk."""
        self.index_sampled = random.sample(range(0, len(self.paths_H), 1), self.num_sampled)
        n_count = 0

        for i in range(len(self.index_sampled)):
            L_patches, H_patches = self.get_patches(self.index_sampled[i])
            for (L_patch, H_patch) in zip(L_patches, H_patches):
                self.L_data[n_count, :, :, :] = L_patch
                self.H_data[n_count, :, :, :] = H_patch
                n_count += 1

        # H/L patch buffers are filled above. No progress print: this runs inside
        # the automation loop where log noise (and stdout coupling) is unwanted.

    def get_patches(self, index):
        """Extract ``num_patches_per_image`` random L/H patch pairs from the image pair at ``index``."""
        L_path = self.paths_L[index]
        H_path = self.paths_H[index]
        img_L = util.imread_uint(L_path, self.n_channels)  # uint format
        img_H = util.imread_uint(H_path, self.n_channels)  # uint format

        H, W = img_H.shape[:2]

        L_patches, H_patches = [], []

        num = self.num_patches_per_image
        for _ in range(num):
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            L_patch = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            H_patch = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            L_patches.append(L_patch)
            H_patches.append(H_patch)

        return L_patches, H_patches

    def _make_sample(self, img_H, img_L, index):
        """Augment the paired H/L patch (train flips/rotates both with the same mode); test returns them unchanged."""
        if self.opt['phase'] == 'train':
            mode = random.randint(0, 7)
            img_L = util.augment_img(img_L, mode=mode)
            img_H = util.augment_img(img_H, mode=mode)
        return img_H, img_L

    def __getitem__(self, index):
        """Return ``{'L', 'H'}`` as float32 tensors (uint8->tensor)."""
        if self.opt['phase'] == 'train':
            patch_L = self.L_data[index]
            patch_H = self.H_data[index]

            patch_H, patch_L = self._make_sample(patch_H, patch_L, index)

            patch_L, patch_H = util.uint2tensor3(patch_L), util.uint2tensor3(patch_H)
        else:
            L_path = self.paths_L[index]
            H_path = self.paths_H[index]
            patch_L = util.imread_uint(L_path, self.n_channels)
            patch_H = util.imread_uint(H_path, self.n_channels)

            patch_H, patch_L = self._make_sample(patch_H, patch_L, index)

            patch_L, patch_H = util.uint2tensor3(patch_L), util.uint2tensor3(patch_H)

        return {'L': patch_L, 'H': patch_H}

    def __len__(self):
        return self.total_patches
