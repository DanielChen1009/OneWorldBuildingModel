from model import AttentionUNET
import torch
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import torchvision
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_ID = 11024


def main():
    # mask = np.array(Image.open(mask_path).convert("L"))
    # print("=> Saving Mask")
    # torchvision.utils.save_image(
    #     mask.unsequeeze(1), "testing/mask.png"
    # )
    # print("=> Mask Saved")
    train_transform = A.Compose(
        [
            A.Resize(height=240, width=320),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    mask = np.array(Image.open(f"data/test/test-label-img/{IMAGE_ID}_lab.png").convert("L"), dtype=np.float32)
    image = np.array(Image.open(f"data/test/test-org-img/{IMAGE_ID}.jpg").convert("RGB"))

    alb = train_transform(image=image, mask=mask)
    image = alb["image"]
    mask = alb["mask"]
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
    mask = mask / 4

    torchvision.utils.save_image(
        mask, "testing/mask.png"
    )

    model = AttentionUNET(in_channels=3, out_channels=1)
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    with torch.no_grad():
        pred = torch.sigmoid(model(image.unsqueeze(0)))
        pred = torch.where(np.logical_and(pred >= 0, pred < 0.2), torch.mul(torch.ones(pred.size()), 0),
                           pred)
        pred = torch.where(np.logical_and(pred >= 0.2, pred < 0.4), torch.mul(torch.ones(pred.size()), 0.25),
                           pred)
        pred = torch.where(np.logical_and(pred >= 0.4, pred < 0.6), torch.mul(torch.ones(pred.size()), 0.5),
                           pred)
        pred = torch.where(np.logical_and(pred >= 0.6, pred < 0.8), torch.mul(torch.ones(pred.size()), 0.75),
                           pred)
        pred = torch.where(np.logical_and(pred >= 0.8, pred <= 1), torch.mul(torch.ones(pred.size()), 1.0),
                           pred)
    print("=> Saving Pred")
    torchvision.utils.save_image(
        pred, "testing/pred.png"
    )
    print("=> Pred Saved")


if __name__ == "__main__":
    main()
