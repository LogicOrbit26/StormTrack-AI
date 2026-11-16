import os
import pandas as pd

BASE_DIR = r"C:\StormTrack-AI\ml_engine"

PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "real_gonu",
    "gonu_label_match_preview.csv"
)

df = pd.read_csv(PATH)

df["observation_time"] = pd.to_datetime(
    df["observation_time"],
    utc=True
)

# Select Meteosat-7 observations
met7 = df[df["platform"] == "Meteosat-7"].copy()

print("Total Meteosat-7 images:", len(met7))

print(
    "Unique Meteosat-7 timestamps:",
    met7["observation_time"].nunique()
)

print(
    "Duplicate Meteosat-7 timestamps:",
    met7["observation_time"].duplicated().sum()
)

all_times = set(df["observation_time"].unique())
met7_times = set(met7["observation_time"].unique())

print(
    "Timestamps covered by Meteosat-7:",
    len(met7_times)
)

print(
    "Timestamps missing from Meteosat-7:",
    len(all_times - met7_times)
)

print("\nPlatform coverage check:")

if (
    len(met7) == 63
    and met7["observation_time"].nunique() == 63
    and all_times == met7_times
):
    print("Meteosat-7 covers all 63 timestamps exactly once.")
else:
    print("Meteosat-7 does not cover every timestamp exactly once.")