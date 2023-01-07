import numpy as np
import torch
import torchvision
from dataset import DisasterDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchmetrics.functional import dice


def save_checkpoint(state, filename="checkpoint_CrossEnt.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print("=> Checkpoint Saved")


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint)
    print("=> Checkpoint Loaded")


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = DisasterDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        limit=2000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = DisasterDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        limit=500
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, folder="saved_images/", device="cuda"):
    print("=> Checking Accuracy")
    num_correct = 0
    num_pixels = 0
    model.eval()
    sum_score = 0
    num_zeros = 0
    ind = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            soft_max = nn.Softmax(dim=1)
            single_dim_preds = soft_max(preds)
            single_dim_preds = torch.argmax(single_dim_preds, dim=1)
            num_correct += (single_dim_preds == y).sum()
            num_zeros += (single_dim_preds == torch.zeros(single_dim_preds.size()).float()).sum()
            num_pixels += torch.numel(single_dim_preds)
            print(ind, end=' ')
            sum_score += dice(preds, y.int()).item()
            torchvision.utils.save_image(
                torch.div(single_dim_preds, 4).unsqueeze(1), f"{folder}pred_{ind}.png"
            )
            torchvision.utils.save_image(torch.div(y.unsqueeze(1), 4), f"{folder}{ind}.png")
            ind += 1
        print()
    num_correct -= num_zeros
    num_pixels -= num_zeros
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {sum_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    print("=> Saving Predictions")
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        torchvision.utils.save_image(
            x, f"{folder}orig_{idx}.png"
        )
        with torch.no_grad():
            preds = model(x)
            soft_max = nn.Softmax(dim=1)
            preds = soft_max(preds)
            preds = torch.argmax(preds, dim=1)
            preds = torch.div(preds, 4)
        print(idx, end=' ')
        torchvision.utils.save_image(
            preds.unsqueeze(1), f"{folder}pred_{idx}.png"
        )
        torchvision.utils.save_image(torch.div(y.unsqueeze(1), 4), f"{folder}{idx}.png")
    print()
    print("=> Images Saved")

    model.train()
