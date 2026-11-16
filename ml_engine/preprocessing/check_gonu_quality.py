
import os
import numpy as np
import xarray as xr

DATA_DIR = r"C:\StormTrack-AI\ml_engine\dataset\real\satellite\GONU"

files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".nc")
)

total_pixels = 0
total_missing = 0
valid_pixels = 0
invalid_files = []

for filename in files:
    path = os.path.join(DATA_DIR, filename)

    with xr.open_dataset(path) as ds:
        values = ds["IRWIN"].values.astype(float)

        total_pixels += values.size
        missing = np.isnan(values).sum()
        total_missing += missing

        valid = np.isfinite(values)
        valid_pixels += valid.sum()

        if not valid.any():
            invalid_files.append(filename)

print("Total files:", len(files))
print("Total pixels:", total_pixels)
print("Missing pixels:", total_missing)
print("Valid pixels:", valid_pixels)

print(
    "Missing percentage:",
    round(total_missing / total_pixels * 100, 3), "%"
)

print("Files without valid observations:", invalid_files)
