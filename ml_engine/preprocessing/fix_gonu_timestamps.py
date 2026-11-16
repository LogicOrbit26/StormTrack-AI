
import os
import csv
import xarray as xr

BASE_DIR = r"C:\StormTrack-AI\ml_engine"

DATA_DIR = os.path.join(
    BASE_DIR, "dataset", "real", "satellite", "GONU"
)

CSV_PATH = os.path.join(
    BASE_DIR, "dataset", "processed", "real_gonu", "metadata.csv"
)

with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for i, row in enumerate(rows):
    path = os.path.join(DATA_DIR, row["filename"])

    with xr.open_dataset(
        path,
        engine="netcdf4",
        decode_cf=False
    ) as ds:

        time_var = ds["htime"]

        decoded = xr.coding.times.decode_cf_datetime(
            time_var.values,
            units=time_var.attrs["units"],
            calendar=time_var.attrs.get("calendar", "standard")
        )

        row["observation_time"] = str(decoded[0]) + " UTC"

    if (i + 1) % 20 == 0:
        print(f"Updated {i + 1}/{len(rows)} timestamps")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nTimestamp correction completed!")
print("First observation:", rows[0]["observation_time"])
print("Metadata rows updated:", len(rows))
