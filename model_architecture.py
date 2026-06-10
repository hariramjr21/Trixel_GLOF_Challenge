import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ==========================================
# BUILD MODEL
# ==========================================

def build_model():

    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    return model


# ==========================================
# LOSS FUNCTION
# ==========================================

def get_loss_function():

    bce_loss = nn.BCEWithLogitsLoss()

    tversky_loss = smp.losses.TverskyLoss(
        mode="binary",
        alpha=0.7,
        beta=0.3
    )

    def criterion(logits, targets):

        targets = targets.float().unsqueeze(1)

        bce = bce_loss(
            logits,
            targets
        )

        tv = tversky_loss(
            logits,
            targets
        )

        return 0.5 * bce + 0.5 * tv

    return criterion


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_trained_model(
    model_path,
    device="cpu"
):

    model = build_model()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model


# ==========================================
# MODEL SUMMARY
# ==========================================

if __name__ == "__main__":

    model = build_model()

    print(model)