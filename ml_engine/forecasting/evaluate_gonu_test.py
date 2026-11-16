
import os
import sys
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

CHECKPOINT_PATH = os.path.join(
    ML_DIR,
    "weights",
    "gonu_cyclone_resnet18.pth"
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== GONU TEST EVALUATION ==========")
print("Device:", device)

# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model = CycloneIntensityModel(
    pretrained=False
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

target_mean = checkpoint["target_mean"].to(device)
target_std = checkpoint["target_std"].to(device)

print("Trained checkpoint loaded successfully!")
print("Best epoch:", checkpoint["best_epoch"])

# --------------------------------------------------
# LOAD TEST DATASET ONLY
# --------------------------------------------------

test_dataset = GONUDataset("test")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test observations:", len(test_dataset))

# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

criterion = nn.MSELoss()

all_predictions = []
all_targets = []
total_loss = 0.0

with torch.no_grad():

    for images, masks, targets in test_loader:

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

        # Convert predictions to original units
        predictions = (
            predictions * target_std + target_mean
        )

        all_predictions.append(
            predictions.cpu().numpy()
        )

        all_targets.append(
            targets.cpu().numpy()
        )

predictions = np.concatenate(all_predictions)
actual = np.concatenate(all_targets)

errors = predictions - actual

# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

wind_mae = np.mean(np.abs(errors[:, 0]))
pressure_mae = np.mean(np.abs(errors[:, 1]))

wind_rmse = np.sqrt(np.mean(errors[:, 0] ** 2))
pressure_rmse = np.sqrt(np.mean(errors[:, 1] ** 2))

average_loss = total_loss / len(test_loader)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n========== FINAL TEST RESULTS ==========")

print("Test observations:", len(test_dataset))
print("Best training epoch:", checkpoint["best_epoch"])
print("Test loss:", round(average_loss, 4))

print("\nWIND SPEED")
print("MAE:", round(wind_mae, 2), "knots")
print("RMSE:", round(wind_rmse, 2), "knots")

print("\nATMOSPHERIC PRESSURE")
print("MAE:", round(pressure_mae, 2), "mb")
print("RMSE:", round(pressure_rmse, 2), "mb")

print("\n========== INDIVIDUAL PREDICTIONS ==========")

for i in range(len(actual)):

    print(f"\nObservation {i + 1}")

    print(
        "Actual wind:", round(float(actual[i, 0]), 2),
        "| Predicted wind:", round(float(predictions[i, 0]), 2)
    )

    print(
        "Actual pressure:", round(float(actual[i, 1]), 2),
        "| Predicted pressure:", round(float(predictions[i, 1]), 2)
    )

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results = np.column_stack((
    actual[:, 0],
    predictions[:, 0],
    np.abs(errors[:, 0]),
    actual[:, 1],
    predictions[:, 1],
    np.abs(errors[:, 1])
))

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gonu_test_results.csv"
)

np.savetxt(
    output_path,
    results,
    delimiter=",",
    header=(
        "actual_wind,predicted_wind,wind_error,"
        "actual_pressure,predicted_pressure,pressure_error"
    ),
    comments="",
    fmt="%.4f"
)

print("\nResults saved to:")
print(output_path)

print("\nTEST EVALUATION COMPLETED!")
