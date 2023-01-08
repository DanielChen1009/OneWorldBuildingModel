import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T


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
        mask[not3 & not4 & not5 & not6] = 0.0
        mask[mask == 3.0] = 1.0
        mask[mask == 4.0] = 2.0
        mask[mask == 5.0] = 3.0
        mask[mask == 6.0] = 4.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask


if __name__ == '__main__':
    pass
    # train_transform = A.Compose(
    #     [
    #         A.Resize(height=240, width=320),
    #         A.Rotate(limit=35, p=1.0),
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.1),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )
    # transform = T.ToPILImage()
    # image = np.array(Image.open('data/train/train-org-img/10778.jpg').convert("RGB"))
    # mask = np.array(Image.open('data/train/train-label-img/10778_lab.png').convert("L"), dtype=np.float32)
    # img = transform(torch.div(torch.from_numpy(mask), 11))
    # img.show()
    # for line in mask:
    #     print(line)
    # not3 = mask != 3.0
    # not4 = mask != 4.0
    # not5 = mask != 5.0
    # not6 = mask != 6.0
    # mask[not3 & not4 & not5 & not6] = 0.0
    # mask[mask == 3.0] = 1.0
    # mask[mask == 4.0] = 2.0
    # mask[mask == 5.0] = 3.0
    # mask[mask == 6.0] = 4.0
    #
    # # for line in mask:
    # #     print(line)
    #
    #
    # augs = train_transform(image=image, mask=mask)
    #
    # img = transform(augs['image'])
    # # img.show()
    # img = transform(torch.div(torch.from_numpy(mask), 11))
    # img.show()
    # img = transform(torch.div(augs['mask'], 4))
    # print(np.max(mask))
    # print(np.min(mask))
    #
