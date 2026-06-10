import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    cohen_kappa_score
)


# ==========================================
# CREATE OVERLAY
# ==========================================

def create_overlay(image, mask):

    overlay = image.copy()

    overlay[mask == 1] = [0, 255, 255]

    return overlay


# ==========================================
# MASK TO BINARY
# ==========================================

def mask_to_binary(mask):

    return (mask > 0).astype(np.uint8)


# ==========================================
# LAKE STATISTICS
# ==========================================

def lake_statistics(mask):

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lake_count = len(contours)

    areas = [
        cv2.contourArea(c)
        for c in contours
    ]

    total_area = sum(areas)

    largest_lake = max(areas) if len(areas) > 0 else 0

    return lake_count, total_area, largest_lake


# ==========================================
# EVALUATION METRICS
# ==========================================

def calculate_metrics(
    y_true,
    y_pred
):

    metrics = {

        "Accuracy":
        accuracy_score(
            y_true,
            y_pred
        ),

        "Precision":
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall":
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1":
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "IoU":
        jaccard_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Kappa":
        cohen_kappa_score(
            y_true,
            y_pred
        )

    }

    return metrics


# ==========================================
# SAVE OUTPUTS
# ==========================================

def save_results(
    mask,
    overlay,
    output_folder="outputs"
):

    import os

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    cv2.imwrite(
        os.path.join(
            output_folder,
            "predicted_mask.png"
        ),
        mask * 255
    )

    cv2.imwrite(
        os.path.join(
            output_folder,
            "overlay.png"
        ),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )