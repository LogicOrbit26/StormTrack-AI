
import os
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = r"C:\StormTrack-AI\ml_engine"

METADATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "real_gonu",
    "metadata.csv"
)

IBTRACS_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real",
    "labels",
    "ibtracs.NI.list.v04r01.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "real_gonu",
    "gonu_label_match_preview.csv"
)

# --------------------------------------------------
# STEP 1: LOAD SATELLITE METADATA
# --------------------------------------------------

print("Loading satellite metadata...")

sat = pd.read_csv(METADATA_PATH)

sat["observation_time"] = pd.to_datetime(
    sat["observation_time"],
    utc=True
).astype("datetime64[ns, UTC]")

print("Satellite observations:", len(sat))

# --------------------------------------------------
# STEP 2: LOAD IBTRACS DATA
# --------------------------------------------------

print("\nLoading IBTrACS data...")

ib = pd.read_csv(
    IBTRACS_PATH,
    skiprows=[1],
    low_memory=False
)

# Select only GONU records
ib = ib[
    ib["SID"] == "2007151N14072"
].copy()

ib["ISO_TIME"] = pd.to_datetime(
    ib["ISO_TIME"],
    utc=True
).astype("datetime64[ns, UTC]")

# Select required columns
ib = ib[
    ["ISO_TIME", "USA_WIND", "USA_PRES"]
].copy()

# Convert intensity labels into numeric values
ib["USA_WIND"] = pd.to_numeric(
    ib["USA_WIND"],
    errors="coerce"
)

ib["USA_PRES"] = pd.to_numeric(
    ib["USA_PRES"],
    errors="coerce"
)

print("IBTrACS GONU records:", len(ib))

# --------------------------------------------------
# STEP 3: CHECK DUPLICATE TIMESTAMPS
# --------------------------------------------------

duplicate_count = ib["ISO_TIME"].duplicated().sum()

print(
    "Duplicate IBTrACS timestamps:",
    duplicate_count
)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate timestamps found. "
        "Please investigate before matching."
    )

# --------------------------------------------------
# STEP 4: SORT DATASETS
# --------------------------------------------------

sat = sat.sort_values(
    "observation_time"
).reset_index(drop=True)

ib = ib.sort_values(
    "ISO_TIME"
).reset_index(drop=True)

# --------------------------------------------------
# STEP 5: MATCH SATELLITE AND IBTRACS RECORDS
# --------------------------------------------------

print("\nMatching satellite observations with IBTrACS...")

matched = pd.merge_asof(
    sat,
    ib,
    left_on="observation_time",
    right_on="ISO_TIME",
    direction="nearest",
    tolerance=pd.Timedelta("90 minutes")
)

# Calculate absolute time difference
matched["time_difference_minutes"] = (
    matched["observation_time"] - matched["ISO_TIME"]
).abs().dt.total_seconds() / 60

# --------------------------------------------------
# STEP 6: MATCHING SUMMARY
# --------------------------------------------------

matched_count = matched["ISO_TIME"].notna().sum()

unmatched_count = matched["ISO_TIME"].isna().sum()

exact_count = (
    matched["time_difference_minutes"] == 0
).sum()

missing_wind = matched["USA_WIND"].isna().sum()

missing_pressure = matched["USA_PRES"].isna().sum()

print("\n========== MATCHING SUMMARY ==========")

print("Total satellite observations:", len(matched))

print("Exact timestamp matches:", exact_count)

print("Matches within 90 minutes:", matched_count)

print("Unmatched observations:", unmatched_count)

print("Missing USA_WIND labels:", missing_wind)

print("Missing USA_PRES labels:", missing_pressure)

# --------------------------------------------------
# STEP 7: TIME DIFFERENCE ANALYSIS
# --------------------------------------------------

print("\nTime difference distribution (minutes):")

if matched["time_difference_minutes"].notna().any():

    print(
        matched["time_difference_minutes"].describe()
    )

else:

    print("No observations were matched.")

# --------------------------------------------------
# STEP 8: DISPLAY FIRST 10 MATCHES
# --------------------------------------------------

print("\nFirst 10 matched records:")

preview_columns = [
    "observation_time",
    "ISO_TIME",
    "time_difference_minutes",
    "USA_WIND",
    "USA_PRES"
]

print(
    matched[preview_columns]
    .head(10)
    .to_string(index=False)
)

# --------------------------------------------------
# STEP 9: SAVE MATCHING PREVIEW
# --------------------------------------------------

matched.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nPreview saved successfully!")

print("Output file:")
print(OUTPUT_PATH)

print("\n======================================")
