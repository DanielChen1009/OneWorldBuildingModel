import cv2

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
import torch.nn as nn

IMAGE_ID = 11290
interpol = cv2.INTER_AREA


def main():
    # mask = np.array(Image.open(mask_path).convert("L"))
    # print("=> Saving Mask")
    # torchvision.utils.save_image(
    #     mask.unsequeeze(1), "testing/mask.png"
    # )
    # print("=> Mask Saved")
    train_transform = A.Compose(
        [
            A.Resize(height=240, width=240, interpolation=interpol),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    mask = np.array(Image.open(f"data/rescuenet/test/test-label-img/{IMAGE_ID}_lab.png").convert("L"), dtype=np.float32)
    # image = np.array(Image.open(f"data/test/test-org-img/{IMAGE_ID}.jpg").convert("RGB"))
    image = np.array(Image.open(f"analysis.png"))
    alb = train_transform(image=image, mask=mask)
    image = alb["image"]
    mask = alb["mask"]
    not3 = mask != 3.0
    not4 = mask != 4.0
    not5 = mask != 5.0
    not6 = mask != 6.0
    mask[not3 & not4 & not5 & not6] = 0.0
    mask[mask == 3.0] = 1.0
    mask[mask == 4.0] = 2.0
    mask[mask == 5.0] = 3.0
    mask[mask == 6.0] = 4.0
    mask = mask / 4
    print(mask.unsqueeze(0).size())
    mask = mask.unsqueeze(0)
    mask = torchvision.transforms.Resize((3000, 4000), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(mask)
    torchvision.utils.save_image(
        mask, "testing/mask.png"
    )
    torchvision.utils.save_image(
        image, 'testing/testing.png'
    )

    model = AttentionUNET(in_channels=3, out_channels=5)
    load_checkpoint(torch.load("checkpoint_FocalLoss.pth.tar")['state_dict'], model)
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(image.unsqueeze(0)))
        soft_max = nn.Softmax(dim=1)
        single_dim_preds = soft_max(pred)
        single_dim_preds = torch.argmax(single_dim_preds, dim=1).float()
        # single_dim_preds = torchvision.transforms.Resize((320, 320), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(single_dim_preds)
    print("=> Saving Pred")
    torchvision.utils.save_image(
        torch.div(single_dim_preds, 4), "testing/pred.png"
    )
    print("=> Pred Saved")
    model.train()


if __name__ == "__main__":
    main()
