import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model_architecture import (
    build_model,
    get_loss_function
)

# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_DIR = "images"
MASK_DIR = "masks"

IMG_SIZE = 512

BATCH_SIZE = 4

EPOCHS = 50

LR = 1e-4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================
# DATASET
# ==========================================

class LakeDataset(Dataset):

    def __init__(
        self,
        files,
        transform
    ):

        self.files = files

        self.transform = transform

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]

        image = cv2.imread(
            os.path.join(
                IMAGE_DIR,
                file
            )
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            os.path.join(
                MASK_DIR,
                file
            ),
            0
        )

        mask = (
            mask > 0
        ).astype(np.float32)

        aug = self.transform(
            image=image,
            mask=mask
        )

        return (
            aug["image"],
            aug["mask"]
        )

# ==========================================
# AUGMENTATION
# ==========================================

train_tf = A.Compose([

    A.Resize(
        IMG_SIZE,
        IMG_SIZE
    ),

    A.HorizontalFlip(
        p=0.5
    ),

    A.VerticalFlip(
        p=0.5
    ),

    A.RandomRotate90(
        p=0.5
    ),

    A.Normalize(),

    ToTensorV2()

])

val_tf = A.Compose([

    A.Resize(
        IMG_SIZE,
        IMG_SIZE
    ),

    A.Normalize(),

    ToTensorV2()

])

# ==========================================
# FILE LIST
# ==========================================

files = sorted([

    f for f in os.listdir(
        IMAGE_DIR
    )

    if f.endswith(
        ".png"
    )

])

train_files, val_files = train_test_split(

    files,

    test_size=0.15,

    random_state=42

)

print(
    "Train:",
    len(train_files)
)

print(
    "Validation:",
    len(val_files)
)

# ==========================================
# DATA LOADERS
# ==========================================

train_ds = LakeDataset(
    train_files,
    train_tf
)

val_ds = LakeDataset(
    val_files,
    val_tf
)

train_loader = DataLoader(

    train_ds,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=2

)

val_loader = DataLoader(

    val_ds,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2

)

# ==========================================
# MODEL
# ==========================================

model = build_model()

model.to(DEVICE)

criterion = get_loss_function()

optimizer = AdamW(

    model.parameters(),

    lr=LR

)

scheduler = CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS

)

# ==========================================
# TRAINING LOOP
# ==========================================

best_loss = 999999

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for images, masks in tqdm(
        train_loader
    ):

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = criterion(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(
        train_loader
    )

    # ==========================
    # VALIDATION
    # ==========================

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks
            )

            val_loss += loss.item()

    val_loss /= len(
        val_loader
    )

    scheduler.step()

    print(

        f"Epoch {epoch+1}/{EPOCHS}"

        f" | Train Loss: {train_loss:.4f}"

        f" | Val Loss: {val_loss:.4f}"

    )

    # ==========================
    # SAVE BEST MODEL
    # ==========================

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(

            model.state_dict(),

            "best_glacial_model.pth"

        )

        print(
            "Best Model Saved"
        )

print(
    "\nTraining Complete"
)