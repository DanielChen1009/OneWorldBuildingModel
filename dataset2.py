import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T


class XViewDataset(Dataset):

    def __init__(self, image_dir, mask_dir, start, end=None, step=1, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        if end is None:
            end = len(os.listdir(image_dir)) // step
        self.images = os.listdir(image_dir)[start*step:end*step:step]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        mask_path = os.path.join(self.mask_dir, self.images[item].replace(".png", "_target.png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
