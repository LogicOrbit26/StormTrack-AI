
import os
import numpy as np
import xarray as xr

BASE = r"C:\StormTrack-AI\ml_engine"
DATA_DIR = os.path.join(BASE, "dataset", "real", "satellite", "GONU")
MASK_DIR = os.path.join(BASE, "dataset", "processed", "real_gonu", "masks")

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".nc"))

total_valid = 0
total_invalid = 0
zero_invalid_files = 0

for filename in files:
    mask_name = os.path.splitext(filename)[0] + "_mask.npy"
    mask_path = os.path.join(MASK_DIR, mask_name)

    mask = np.load(mask_path)

    valid = int(np.sum(mask == 1))
    invalid = int(np.sum(mask == 0))

    total_valid += valid
    total_invalid += invalid

    if invalid == 0:
        zero_invalid_files += 1

print("Total files:", len(files))
print("Total valid pixels:", total_valid)
print("Total invalid pixels:", total_invalid)
print("Files with zero invalid pixels:", zero_invalid_files)

# Inspect time metadata from the first NetCDF file
first_file = os.path.join(DATA_DIR, files[0])

with xr.open_dataset(
    first_file,
    engine="netcdf4",
    decode_cf=False
) as ds:

    print("\nTime variable:")
    print("Value:", ds["htime"].values)
    print("Attributes:", ds["htime"].attrs)

print("\nQuality inspection completed.")
