import sys
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model_architecture import load_trained_model
from utils import (
    create_overlay,
    lake_statistics,
    save_results
)

# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "best_glacial_model.pth"

IMG_SIZE = 512

# ==========================================
# TRANSFORM
# ==========================================

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# ==========================================
# LOAD MODEL
# ==========================================

model = load_trained_model(
    MODEL_PATH,
    DEVICE
)

print("Model Loaded Successfully")

# ==========================================
# INPUT IMAGE
# ==========================================

if len(sys.argv) < 2:

    print(
        "Usage: python inference.py image.png"
    )

    exit()

image_path = sys.argv[1]

# ==========================================
# READ IMAGE
# ==========================================

img = cv2.imread(image_path)

if img is None:

    print("Unable to read image")

    exit()

img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

h, w = img_rgb.shape[:2]

# ==========================================
# PREPROCESS
# ==========================================

tensor = transform(
    image=img_rgb
)["image"]

tensor = tensor.unsqueeze(0).to(DEVICE)

# ==========================================
# PREDICT
# ==========================================

with torch.no_grad():

    pred = torch.sigmoid(
        model(tensor)
    )

pred = pred.squeeze().cpu().numpy()

pred = (
    pred > 0.5
).astype(np.uint8)

# ==========================================
# RESIZE TO ORIGINAL SIZE
# ==========================================

mask = cv2.resize(
    pred,
    (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# CREATE OVERLAY
# ==========================================

overlay = create_overlay(
    img_rgb,
    mask
)

# ==========================================
# LAKE STATISTICS
# ==========================================

lake_count, total_area, largest_lake = lake_statistics(mask)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n===== RESULTS =====")

print(
    "Lake Count:",
    lake_count
)

print(
    "Total Lake Area:",
    total_area
)

print(
    "Largest Lake:",
    largest_lake
)

# ==========================================
# SAVE OUTPUTS
# ==========================================

save_results(
    mask,
    overlay
)

print(
    "\nSaved in outputs folder"
)