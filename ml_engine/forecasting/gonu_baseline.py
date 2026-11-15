
import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset", "processed",
    "real_gonu_meteosat7"
)

SPLIT_PATH = os.path.join(
    DATASET_DIR, "splits.csv"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading GONU dataset splits...")

df = pd.read_csv(SPLIT_PATH)

# --------------------------------------------------
# TRAINING DATA
# --------------------------------------------------

train = df[df["split"] == "train"]

train_wind_mean = train["wind_speed_knots"].mean()
train_pressure_mean = train["pressure_mb"].mean()

print("\n========== TRAINING BASELINE ==========")

print("Training observations:", len(train))

print(
    "Mean wind speed:",
    round(train_wind_mean, 2),
    "knots"
)

print(
    "Mean pressure:",
    round(train_pressure_mean, 2),
    "mb"
)

# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_baseline(data, split_name):

    actual_wind = data["wind_speed_knots"].to_numpy()
    actual_pressure = data["pressure_mb"].to_numpy()

    predicted_wind = np.full(
        len(data), train_wind_mean
    )

    predicted_pressure = np.full(
        len(data), train_pressure_mean
    )

    wind_mae = np.mean(
        np.abs(actual_wind - predicted_wind)
    )

    wind_rmse = np.sqrt(
        np.mean((actual_wind - predicted_wind) ** 2)
    )

    pressure_mae = np.mean(
        np.abs(actual_pressure - predicted_pressure)
    )

    pressure_rmse = np.sqrt(
        np.mean((actual_pressure - predicted_pressure) ** 2)
    )

    print(f"\n========== {split_name.upper()} RESULTS ==========")

    print("Observations:", len(data))

    print("Wind MAE:", round(wind_mae, 2), "knots")
    print("Wind RMSE:", round(wind_rmse, 2), "knots")

    print("Pressure MAE:", round(pressure_mae, 2), "mb")
    print("Pressure RMSE:", round(pressure_rmse, 2), "mb")

    return {
        "split": split_name,
        "observations": len(data),
        "wind_mae": wind_mae,
        "wind_rmse": wind_rmse,
        "pressure_mae": pressure_mae,
        "pressure_rmse": pressure_rmse
    }

# --------------------------------------------------
# VALIDATION AND TEST
# --------------------------------------------------

validation = df[df["split"] == "validation"]
test = df[df["split"] == "test"]

results = []

results.append(
    evaluate_baseline(validation, "validation")
)

results.append(
    evaluate_baseline(test, "test")
)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_path = os.path.join(
    DATASET_DIR,
    "baseline_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nBaseline results saved to:")
print(output_path)

print("\nBaseline evaluation completed!")
