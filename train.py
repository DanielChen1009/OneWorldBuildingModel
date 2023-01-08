import gc

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import AttentionUNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
# Perhaps change learning rate...
LEARNING_RATE = 1e-4
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    DEVICE = torch.device("cpu")

else:
    DEVICE = torch.device("mps")
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 1
IMAGE_HEIGHT = 240  # 3000 originally
IMAGE_WIDTH = 320  # 4000 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train/train-org-img"
TRAIN_MASK_DIR = "data/train/train-label-img"
VAL_IMG_DIR = "data/val/val-org-img"
VAL_MASK_DIR = "data/val/val-label-img"
LOSS_SUM = 0


def train_fn(loader, model, optimizer, loss_fn, scaler, loss_sum):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets.long())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # update tqdm loop
        loss_sum += loss.item()
        loop.set_postfix(loss=(loss_sum/(batch_idx + 1)))
    print(loss_sum / 125)


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = AttentionUNET(in_channels=3, out_channels=5).to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(torch.load("checkpoint_CrossEnt.pth.tar", map_location='cpu')["optimizer"])

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint_CrossEnt.pth.tar", map_location='cpu')["state_dict"], model)
    model = model.to(device=DEVICE)
    check_accuracy(val_loader, model, folder="saved_images/", device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    # save_predictions_as_imgs(
    #     val_loader, model, folder="saved_images/", device=DEVICE
    # )

    for epoch in range(NUM_EPOCHS):
        print(optimizer.param_groups[-1]['lr'])
        train_fn(train_loader, model, optimizer, loss_fn, scaler, 0)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE
        # )


if __name__ == "__main__":
    main()
