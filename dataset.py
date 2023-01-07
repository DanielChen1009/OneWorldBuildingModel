import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class DisasterDataset(Dataset):

    def __init__(self, image_dir, mask_dir, limit, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        mask_path = os.path.join(self.mask_dir, self.images[item].replace(".jpg", "_lab.png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        not3 = mask != 3.0
        not4 = mask != 4.0
        not5 = mask != 5.0
        not6 = mask != 6.0
        finalbool = np.logical_and(np.logical_and(not3, not4), np.logical_and(not5, not6))
        mask[finalbool] = 0.0
        mask[mask == 3.0] = 1.0
        mask[mask == 4.0] = 2.0
        mask[mask == 5.0] = 3.0
        mask[mask == 6.0] = 4.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
