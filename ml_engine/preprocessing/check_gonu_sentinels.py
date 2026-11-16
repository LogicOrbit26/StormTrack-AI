
import os
import numpy as np
import xarray as xr

DATA_DIR = r"C:\StormTrack-AI\ml_engine\dataset\real\satellite\GONU"

files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".nc")
)

print("Total files:", len(files))

for filename in files:
    path = os.path.join(DATA_DIR, filename)

    with xr.open_dataset(path, decode_cf=False) as ds:
        var = ds["IRWIN"]
        raw = var.values

        fill = var.attrs.get("_FillValue")
        scale = float(var.attrs.get("scale_factor", 1))
        offset = float(var.attrs.get("add_offset", 0))
        cal_offset = float(var.attrs.get("Cal_Offset", np.nan))

        target = round((cal_offset - offset) / scale)

        fill_count = np.sum(raw == fill) if fill is not None else 0
        target_count = np.sum(raw == target)

        if target_count > 0:
            print(
                filename,
                "| Platform:", ds.attrs.get("Satellite_Name"),
                "| Calibration target:", target,
                "| Target count:", int(target_count),
                "| Fill count:", int(fill_count)
            )
