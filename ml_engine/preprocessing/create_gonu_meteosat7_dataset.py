
import os
import shutil
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = r"C:\StormTrack-AI\ml_engine"

PREVIEW_PATH = os.path.join(
    BASE_DIR,
    "dataset", "processed",
    "real_gonu",
    "gonu_label_match_preview.csv"
)

SOURCE_IMAGES = os.path.join(
    BASE_DIR,
    "dataset", "processed",
    "real_gonu", "images"
)

SOURCE_MASKS = os.path.join(
    BASE_DIR,
    "dataset", "processed",
    "real_gonu", "masks"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset", "processed",
    "real_gonu_meteosat7"
)

OUTPUT_IMAGES = os.path.join(
    OUTPUT_DIR, "images"
)

OUTPUT_MASKS = os.path.join(
    OUTPUT_DIR, "masks"
)

# --------------------------------------------------
# STEP 1: CREATE OUTPUT FOLDERS
# --------------------------------------------------

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_MASKS, exist_ok=True)

# --------------------------------------------------
# STEP 2: LOAD MATCHED DATA
# --------------------------------------------------

print("Loading GONU matching preview...")

df = pd.read_csv(PREVIEW_PATH)

print("Total observations:", len(df))

# --------------------------------------------------
# STEP 3: FILTER METEOSAT-7
# --------------------------------------------------

met7 = df[
    df["platform"] == "Meteosat-7"
].copy()

met7["observation_time"] = pd.to_datetime(
    met7["observation_time"],
    utc=True
)

met7 = met7.sort_values(
    "observation_time"
).reset_index(drop=True)

print("\nMeteosat-7 observations:", len(met7))

# --------------------------------------------------
# STEP 4: VALIDATE OBSERVATIONS
# --------------------------------------------------

if len(met7) != 63:
    raise ValueError(
        f"Expected 63 images, found {len(met7)}."
    )

if met7["observation_time"].duplicated().any():
    raise ValueError(
        "Duplicate timestamps found in Meteosat-7."
    )

if met7["USA_WIND"].isna().any():
    raise ValueError("Missing wind labels found.")

if met7["USA_PRES"].isna().any():
    raise ValueError("Missing pressure labels found.")

# --------------------------------------------------
# STEP 5: COPY IMAGES AND MASKS
# --------------------------------------------------

print("\nCopying images and masks...")

image_files = []
mask_files = []

for index, row in met7.iterrows():

    # Original NetCDF filename
    original_filename = os.path.basename(
        row["filename"]
    )

    # Convert NetCDF filename to NPY image filename
    base_name = os.path.splitext(
        original_filename
    )[0]

    image_name = base_name + ".npy"

    # Mask filename has an additional _mask suffix
    mask_name = base_name + "_mask.npy"

    source_image = os.path.join(
        SOURCE_IMAGES,
        image_name
    )

    source_mask = os.path.join(
        SOURCE_MASKS,
        mask_name
    )

    destination_image = os.path.join(
        OUTPUT_IMAGES,
        image_name
    )

    destination_mask = os.path.join(
        OUTPUT_MASKS,
        mask_name
    )

    # Check source image
    if not os.path.exists(source_image):
        raise FileNotFoundError(
            f"Image not found: {source_image}"
        )

    # Check source mask
    if not os.path.exists(source_mask):
        raise FileNotFoundError(
            f"Mask not found: {source_mask}"
        )

    # Copy image and mask
    shutil.copy2(
        source_image,
        destination_image
    )

    shutil.copy2(
        source_mask,
        destination_mask
    )

    image_files.append(image_name)
    mask_files.append(mask_name)

    if (index + 1) % 10 == 0:
        print(
            f"Copied {index + 1}/{len(met7)} observations"
        )

# --------------------------------------------------
# STEP 6: ADD OUTPUT FILENAMES TO METADATA
# --------------------------------------------------

met7["image_file"] = image_files
met7["mask_file"] = mask_files

# --------------------------------------------------
# STEP 7: SAVE METADATA
# --------------------------------------------------

metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.csv"
)

met7.to_csv(
    metadata_path,
    index=False
)

# --------------------------------------------------
# STEP 8: SAVE INTENSITY LABELS
# --------------------------------------------------

labels = met7[
    [
        "image_file",
        "observation_time",
        "USA_WIND",
        "USA_PRES"
    ]
].copy()

labels = labels.rename(
    columns={
        "USA_WIND": "wind_speed_knots",
        "USA_PRES": "pressure_mb"
    }
)

labels_path = os.path.join(
    OUTPUT_DIR,
    "labels.csv"
)

labels.to_csv(
    labels_path,
    index=False
)

# --------------------------------------------------
# STEP 9: FINAL VERIFICATION
# --------------------------------------------------

saved_images = [
    f for f in os.listdir(OUTPUT_IMAGES)
    if f.endswith(".npy")
]

saved_masks = [
    f for f in os.listdir(OUTPUT_MASKS)
    if f.endswith("_mask.npy")
]

print("\n========== DATASET SUMMARY ==========")

print("Meteosat-7 observations:", len(met7))
print("Saved images:", len(saved_images))
print("Saved masks:", len(saved_masks))
print("Unique timestamps:", met7["observation_time"].nunique())
print("Missing wind labels:", labels["wind_speed_knots"].isna().sum())
print("Missing pressure labels:", labels["pressure_mb"].isna().sum())

print("\nOutput directory:")
print(OUTPUT_DIR)

if (
    len(saved_images) == 63
    and len(saved_masks) == 63
    and len(labels) == 63
):
    print("\nSUCCESS: Meteosat-7 dataset created!")
else:
    print("\nWARNING: Dataset verification failed.")

print("=====================================")
