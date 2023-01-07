from PIL import Image
import torch
import numpy as np
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
# im = Image.open('./data/train/train-label-img/10780_lab.png')

preds = torch.rand(2, 5, 10, 10)
soft_max = nn.Softmax(dim=1)
preds = soft_max(preds)
preds = torch.argmax(preds, dim=1)
print(preds)
print(preds.size())
#
# train_transform = A.Compose(
#     [
#         A.Resize(height=150, width=200),
#         A.Normalize(
#             mean=[0.0, 0.0, 0.0],
#             std=[1.0, 1.0, 1.0],
#             max_pixel_value=10.0,
#         ),
#         ToTensorV2(),
#     ],
# )
#
# mask = np.array(Image.open("data/val/val-label-img/10781_lab.png").convert("L"), dtype=np.float32)
# print(mask)
# image = np.array(Image.open("data/val/val-org-img/10781.jpg").convert("RGB"))
#
# alb = train_transform(image=image, mask=mask)
# image = alb["image"]
# mask = alb["mask"]
# not3 = mask != 3.0
# not4 = mask != 4.0
# not5 = mask != 5.0
# not6 = mask != 6.0
# finalbool = np.logical_and(np.logical_and(not3, not4), np.logical_and(not5, not6))
# mask[finalbool] = 0.0
# mask[mask == 3.0] = 1.0
# mask[mask == 4.0] = 2.0
# mask[mask == 5.0] = 3.0
# mask[mask == 6.0] = 4.0
# mask = mask / 4
# print(mask)
# torchvision.utils.save_image(
#     image, "testing/testing.png"
# )
# torchvision.utils.save_image(
#     mask, "testing/testing_mask.png"
# )
#
# # preds = torch.rand((100, 100))
# # preds = torch.where(np.logical_and(preds >= 0, preds < 0.2), torch.mul(torch.ones(preds.size()), 0), preds)
# # preds = torch.where(np.logical_and(preds >= 0.2, preds < 0.4), torch.mul(torch.ones(preds.size()), 0.25), preds)
# # preds = torch.where(np.logical_and(preds >= 0.4, preds < 0.6), torch.mul(torch.ones(preds.size()), 0.5), preds)
# # preds = torch.where(np.logical_and(preds >= 0.6, preds < 0.8), torch.mul(torch.ones(preds.size()), 0.75), preds)
# # preds = torch.where(np.logical_and(preds >= 0.8, preds <= 1), torch.mul(torch.ones(preds.size()), 1.0), preds)
