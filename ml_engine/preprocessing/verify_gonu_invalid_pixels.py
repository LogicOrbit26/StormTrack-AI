
import os
import numpy as np
import xarray as xr

DATA_DIR = r"C:\StormTrack-AI\ml_engine\dataset\real\satellite\GONU"

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".nc"))

total_pixels = 0
total_fill = 0
total_low = 0
total_calibration = 0

for i, filename in enumerate(files, start=1):
    path = os.path.join(DATA_DIR, filename)

    with xr.open_dataset(
        path,
        engine="netcdf4",
        decode_cf=True,
        drop_variables=[
            name for name in xr.open_dataset(
                path, engine="netcdf4", decode_cf=False
            ).variables if name != "IRWIN"
        ]
    ) as ds:

        values = ds["IRWIN"].values.astype(np.float64)
        cal_offset = float(ds["IRWIN"].attrs["Cal_Offset"])

        total_pixels += values.size
        total_fill += np.isnan(values).sum()
        total_low += np.sum(values < 150)
        total_calibration += np.sum(
            np.isclose(values, cal_offset, atol=0.002)
        )

    if i % 20 == 0:
        print(f"Processed {i}/{len(files)} files")

print("\n--- FINAL RESULTS ---")
print("Total files:", len(files))
print("Total pixels:", total_pixels)
print("NaN pixels:", total_fill)
print("Pixels below 150 K:", total_low)
print("Pixels matching calibration offset:", total_calibration)
