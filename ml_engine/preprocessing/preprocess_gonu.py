
import os
import csv
import numpy as np
import xarray as xr
from PIL import Image

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = r"C:\StormTrack-AI\ml_engine"

INPUT_DIR = os.path.join(
    BASE_DIR, "dataset", "real", "satellite", "GONU"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "dataset", "processed", "real_gonu"
)

IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MIN_TEMP = 180.0
MAX_TEMP = 310.0

# Calibration values within this tolerance of the
# calibration offset are treated as invalid.
CALIBRATION_TOLERANCE = 0.02

files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".nc")
)

metadata = []

# --------------------------------------------------
# PROCESS FILES
# --------------------------------------------------

for index, filename in enumerate(files, start=1):

    path = os.path.join(INPUT_DIR, filename)

    with xr.open_dataset(
        path,
        engine="netcdf4",
        decode_cf=False
    ) as ds:

        variable = ds["IRWIN"]

        raw = variable.values.astype(np.float64).squeeze()

        attrs = variable.attrs

        scale = float(attrs["scale_factor"])
        offset = float(attrs["add_offset"])

        cal_offset = float(attrs["Cal_Offset"])
        cal_scale = float(attrs["Cal_Scale"])

        fill_value = attrs.get("_FillValue")

        # Convert raw values using CF scale and offset.
        corrected = raw * scale + offset

        # --------------------------------------------------
        # VALIDITY MASK
        # --------------------------------------------------

        valid = np.isfinite(raw)

        if fill_value is not None:
            valid &= raw != float(fill_value)

        # HURSAT metadata identifies -1 as a missing value.
        valid &= raw != -1

        # Exclude calibration-offset sentinel values.
        valid &= (
            np.abs(corrected - cal_offset)
            > CALIBRATION_TOLERANCE
        )

        # --------------------------------------------------
        # APPLY CALIBRATION EQUATION
        # --------------------------------------------------

        temperature = (
            corrected - cal_offset
        ) / cal_scale

        valid &= np.isfinite(temperature)

        valid_count = int(np.sum(valid))
        invalid_count = int(valid.size - valid_count)

        if valid_count == 0:
            print("Skipping file with no valid pixels:", filename)
            continue

        # --------------------------------------------------
        # NORMALIZATION
        # --------------------------------------------------

        temperature = np.clip(
            temperature,
            MIN_TEMP,
            MAX_TEMP
        )

        # Normalize to range 0–1.
        normalized = (
            temperature - MIN_TEMP
        ) / (MAX_TEMP - MIN_TEMP)

        # Fill invalid pixels with valid-pixel median.
        median_value = float(np.median(normalized[valid]))

        normalized[~valid] = median_value

        # Convert grayscale to 3 channels.
        image = np.repeat(
            normalized[:, :, np.newaxis],
            3,
            axis=2
        ).astype(np.float32)

        # --------------------------------------------------
        # SAVE OUTPUTS
        # --------------------------------------------------

        image_name = os.path.splitext(filename)[0]

        np.save(
            os.path.join(IMAGE_DIR, image_name + ".npy"),
            image
        )

        np.save(
            os.path.join(MASK_DIR, image_name + "_mask.npy"),
            valid.astype(np.uint8)
        )

        png = (normalized * 255).astype(np.uint8)

        Image.fromarray(png).save(
            os.path.join(IMAGE_DIR, image_name + ".png")
        )

        # --------------------------------------------------
        # METADATA
        # --------------------------------------------------

        observation_time = str(ds["htime"].values[0])

        metadata.append({
            "filename": filename,
            "observation_time": observation_time,
            "platform": ds.attrs.get("Satellite_Name", "Unknown"),
            "cal_offset": cal_offset,
            "cal_scale": cal_scale,
            "valid_pixels": valid_count,
            "invalid_pixels": invalid_count,
            "valid_percentage": round(
                valid_count / valid.size * 100, 2
            ),
            "temperature_min": round(
                float(np.min(temperature[valid])), 2
            ),
            "temperature_max": round(
                float(np.max(temperature[valid])), 2
            )
        })

    if index % 10 == 0:
        print(f"Processed {index}/{len(files)} files")

# --------------------------------------------------
# SAVE METADATA CSV
# --------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")

if metadata:
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metadata[0].keys()
        )

        writer.writeheader()
        writer.writerows(metadata)

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========== GONU PREPROCESSING COMPLETE ==========")

print("Total NetCDF files:", len(files))
print("Successfully processed:", len(metadata))
print("Skipped:", len(files) - len(metadata))

print("Images saved:", IMAGE_DIR)
print("Masks saved:", MASK_DIR)
print("Metadata saved:", csv_path)
