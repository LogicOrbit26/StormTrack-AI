
import os
import glob
import csv
import numpy as np
import xarray as xr
from PIL import Image

# Folder paths
BASE_DIR = r"C:\StormTrack-AI\ml_engine"

SATELLITE_DIR = os.path.join(
    BASE_DIR, "dataset", "real", "satellite"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "dataset", "processed", "real_nargis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all FY2-C files
files = sorted(glob.glob(
    os.path.join(SATELLITE_DIR, "*.FY2-C.*.nc")
))

print("Total satellite files found:", len(files))

if not files:
    raise FileNotFoundError("No FY2-C files found!")

metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")

successful = 0
failed = 0

with open(metadata_path, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "filename",
        "observation_time",
        "height",
        "width",
        "valid_pixels",
        "missing_pixels",
        "valid_percentage"
    ])

    for index, file_path in enumerate(files, start=1):

        filename = os.path.basename(file_path)

        print(f"\nProcessing {index}/{len(files)}: {filename}")

        try:
            with xr.open_dataset(file_path) as ds:

                irwin = ds["IRWIN"].squeeze(drop=True)

                temperature = irwin.values.astype(np.float32)

                observation_time = str(ds["htime"].values[0])

                units = irwin.attrs.get("units", "Unknown")

            if temperature.ndim != 2:
                raise ValueError("Unexpected image dimensions")

            valid_mask = np.isfinite(temperature)

            if not valid_mask.any():
                raise ValueError("No valid pixels found")

            valid_pixels = int(valid_mask.sum())
            missing_pixels = int((~valid_mask).sum())

            valid_percentage = (
                valid_pixels / temperature.size
            ) * 100

            # Fill missing pixels with median
            median_value = np.nanmedian(temperature)

            temperature = np.where(
                valid_mask, temperature, median_value
            )

            # Normalize brightness temperature
            temperature = np.clip(temperature, 180, 310)

            normalized = (310 - temperature) / 130
            normalized = np.clip(normalized, 0, 1)

            # Convert to 3 channels
            rgb_image = np.repeat(
                normalized[:, :, np.newaxis], 3, axis=2
            )

            rgb_image = (rgb_image * 255).astype(np.uint8)

            # Save processed files
            base_name = filename.replace(".nc", "")

            np.save(
                os.path.join(OUTPUT_DIR, base_name + ".npy"),
                rgb_image
            )

            np.save(
                os.path.join(
                    OUTPUT_DIR, base_name + "_valid_mask.npy"
                ),
                valid_mask
            )

            Image.fromarray(rgb_image).save(
                os.path.join(OUTPUT_DIR, base_name + ".png")
            )

            writer.writerow([
                filename,
                observation_time,
                temperature.shape[0],
                temperature.shape[1],
                valid_pixels,
                missing_pixels,
                round(valid_percentage, 2)
            ])

            successful += 1

            print("Shape:", rgb_image.shape)
            print("Valid pixels:", valid_pixels)
            print("Valid percentage:", round(valid_percentage, 2))
            print("Units:", units)

        except Exception as error:
            failed += 1
            print("ERROR:", error)

print("\n========== BATCH PROCESSING SUMMARY ==========")
print("Total files:", len(files))
print("Successfully processed:", successful)
print("Failed:", failed)
print("Metadata saved:", metadata_path)
print("Output folder:", OUTPUT_DIR)
