
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# --------------------------------------------------
# PATHS AND IMPORTS
# --------------------------------------------------

ML_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, ML_DIR)

from models.cyclone_cnn import CycloneIntensityModel
from forecasting.gonu_dataset_loader import GONUDataset

WEIGHTS_DIR = os.path.join(ML_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    WEIGHTS_DIR,
    "gonu_cyclone_resnet18.pth"
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.0001
PATIENCE = 7

torch.manual_seed(42)
np.random.seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== TRAINING CONFIGURATION ==========")
print("Device:", device)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

train_dataset = GONUDataset("train")
val_dataset = GONUDataset("validation")
test_dataset = GONUDataset("test")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# NORMALIZE TARGETS USING TRAINING DATA ONLY
# --------------------------------------------------

train_targets = train_dataset.labels[
    ["wind_speed_knots", "pressure_mb"]
].to_numpy(dtype=np.float32)

target_mean = torch.tensor(
    train_targets.mean(axis=0),
    dtype=torch.float32,
    device=device
)

target_std = torch.tensor(
    train_targets.std(axis=0),
    dtype=torch.float32,
    device=device
).clamp(min=1e-6)

print("\n========== TARGET NORMALIZATION ==========")
print("Mean:", target_mean.cpu().numpy())
print("Standard deviation:", target_std.cpu().numpy())

# --------------------------------------------------
# MODEL, LOSS AND OPTIMIZER
# --------------------------------------------------

model = CycloneIntensityModel(
    pretrained=False
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, masks, targets in loader:

            images = images.to(device)
            targets = targets.to(device)

            normalized_targets = (
                targets - target_mean
            ) / target_std

            predictions = model(images)

            loss = criterion(
                predictions,
                normalized_targets
            )

            total_loss += loss.item()

            # Convert predictions back to real units
            predictions = (
                predictions * target_std + target_mean
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                targets.cpu().numpy()
            )

    average_loss = total_loss / len(loader)

    predictions = np.concatenate(all_predictions)
    actual = np.concatenate(all_targets)

    errors = predictions - actual

    wind_mae = np.mean(np.abs(errors[:, 0]))
    pressure_mae = np.mean(np.abs(errors[:, 1]))

    wind_rmse = np.sqrt(np.mean(errors[:, 0] ** 2))
    pressure_rmse = np.sqrt(np.mean(errors[:, 1] ** 2))

    return (
        average_loss,
        wind_mae,
        pressure_mae,
        wind_rmse,
        pressure_rmse
    )

# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0

print("\n========== STARTING TRAINING ==========")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, masks, targets in train_loader:

        images = images.to(device)
        targets = targets.to(device)

        normalized_targets = (
            targets - target_mean
        ) / target_std

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(
            predictions,
            normalized_targets
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    (
        val_loss,
        wind_mae,
        pressure_mae,
        wind_rmse,
        pressure_rmse
    ) = evaluate(val_loader)

    scheduler.step(val_loss)

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        f"\nTraining Loss: {train_loss:.4f}"
        f"\nValidation Loss: {val_loss:.4f}"
        f"\nWind MAE: {wind_mae:.2f} knots"
        f"\nPressure MAE: {pressure_mae:.2f} mb"
        f"\nWind RMSE: {wind_rmse:.2f} knots"
        f"\nPressure RMSE: {pressure_rmse:.2f} mb"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        best_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_state,
                "target_mean": target_mean.cpu(),
                "target_std": target_std.cpu(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch
            },
            CHECKPOINT_PATH
        )

        epochs_without_improvement = 0

        print("Best model checkpoint saved!")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break

# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

print("\n========== FINAL TEST EVALUATION ==========")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

(
    test_loss,
    wind_mae,
    pressure_mae,
    wind_rmse,
    pressure_rmse
) = evaluate(test_loader)

print("\nBest epoch:", checkpoint["best_epoch"])
print("Best validation loss:", checkpoint["best_val_loss"])

print("\nTEST RESULTS")
print("Test loss:", round(test_loss, 4))
print("Wind MAE:", round(wind_mae, 2), "knots")
print("Pressure MAE:", round(pressure_mae, 2), "mb")
print("Wind RMSE:", round(wind_rmse, 2), "knots")
print("Pressure RMSE:", round(pressure_rmse, 2), "mb")

print("\nCheckpoint saved at:")
print(CHECKPOINT_PATH)

print("\nGONU CNN TRAINING COMPLETED!")
