
import os
import csv
from datetime import datetime

BASE_DIR = r"C:\StormTrack-AI\ml_engine"

METADATA_PATH = os.path.join(
    BASE_DIR, "dataset", "processed",
    "real_nargis", "metadata.csv"
)

IBTRACS_PATH = os.path.join(
    BASE_DIR, "dataset", "real", "labels",
    "ibtracs.NI.list.v04r01.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "dataset", "processed",
    "real_nargis", "verified_labels.csv"
)

MAX_DIFFERENCE_HOURS = 3.5

AGENCIES = ["USA", "WMO", "NEWDELHI"]

# Load satellite metadata
with open(METADATA_PATH, "r", newline="") as f:
    metadata = list(csv.DictReader(f))

# Load Nargis IBTrACS records
records = []

with open(IBTRACS_PATH, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    next(reader, None)  # Skip units row

    for row in reader:
        if row.get("SID", "").strip() != "2008117N11090":
            continue

        try:
            record_time = datetime.fromisoformat(
                row["ISO_TIME"].strip().replace("Z", "+00:00")
            ).replace(tzinfo=None)

            records.append((record_time, row))

        except (ValueError, TypeError):
            continue

records.sort(key=lambda item: item[0])

output_rows = []

for image in metadata:

    satellite_time = datetime.fromisoformat(
        image["observation_time"]
    )

    nearest_time, nearest_row = min(
        records,
        key=lambda item: abs(
            (item[0] - satellite_time).total_seconds()
        )
    )

    difference_seconds = abs(
        (nearest_time - satellite_time).total_seconds()
    )

    if difference_seconds > MAX_DIFFERENCE_HOURS * 3600:
        print("Skipped:", image["filename"])
        continue

    result = {
        "filename": image["filename"],
        "observation_time": image["observation_time"],
        "ibtracs_time": nearest_time.isoformat(),
        "time_difference_seconds": round(difference_seconds, 3),
    }

    for agency in AGENCIES:
        result[agency + "_WIND_kts"] = (
            nearest_row.get(agency + "_WIND", "").strip()
        )
        result[agency + "_PRES_mb"] = (
            nearest_row.get(agency + "_PRES", "").strip()
        )

    output_rows.append(result)

# Save verified labels
if output_rows:
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=output_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(output_rows)

print("\n========== LABEL DATASET SUMMARY ==========")
print("Satellite images:", len(metadata))
print("Matched records:", len(output_rows))
print("Output:", OUTPUT_PATH)
print("\nVerified label CSV created!")
