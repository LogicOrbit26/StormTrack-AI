
import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = r"C:\StormTrack-AI\ml_engine"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset", "processed",
    "real_gonu_meteosat7"
)

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")

METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.csv")

# --------------------------------------------------
# STEP 1: LOAD FILES
# --------------------------------------------------

print("Loading Meteosat-7 dataset...")

metadata = pd.read_csv(METADATA_PATH)
labels = pd.read_csv(LABELS_PATH)

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".npy")
]

mask_files = [
    f for f in os.listdir(MASK_DIR)
    if f.endswith("_mask.npy")
]

print("Images found:", len(image_files))
print("Masks found:", len(mask_files))
print("Metadata rows:", len(metadata))
print("Label rows:", len(labels))

# --------------------------------------------------
# STEP 2: CHECK COUNTS
# --------------------------------------------------

assert len(image_files) == 63, "Expected 63 images"
assert len(mask_files) == 63, "Expected 63 masks"
assert len(metadata) == 63, "Expected 63 metadata rows"
assert len(labels) == 63, "Expected 63 label rows"

print("\nFile counts verified!")

# --------------------------------------------------
# STEP 3: CHECK IMAGE-MASK PAIRING
# --------------------------------------------------

print("\nChecking image-mask pairs...")

pair_errors = []

for image_name in image_files:

    mask_name = image_name.replace(
        ".npy",
        "_mask.npy"
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_name
    )

    if not os.path.exists(mask_path):
        pair_errors.append(image_name)

print("Missing mask pairs:", len(pair_errors))

# --------------------------------------------------
# STEP 4: CHECK IMAGE AND MASK ARRAYS
# --------------------------------------------------

print("\nChecking dimensions and pixel values...")

shape_errors = []
invalid_images = []
invalid_masks = []
invalid_mask_values = []

for image_name in image_files:

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    mask_name = image_name.replace(
        ".npy",
        "_mask.npy"
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_name
    )

    if not os.path.exists(mask_path):
        continue

    image = np.load(image_path)
    mask = np.load(mask_path)

    if image.shape != (301, 301, 3):
        shape_errors.append(
            f"Image: {image_name}, Shape: {image.shape}"
        )

    if mask.shape != (301, 301):
        shape_errors.append(
            f"Mask: {mask_name}, Shape: {mask.shape}"
        )

    if not np.isfinite(image).all():
        invalid_images.append(image_name)

    if not np.isfinite(mask).all():
        invalid_masks.append(mask_name)

    unique_values = np.unique(mask)

    if not np.isin(unique_values, [0, 1]).all():
        invalid_mask_values.append(
            f"{mask_name}: {unique_values}"
        )

print("Dimension errors:", len(shape_errors))
print("Images with NaN/Inf:", len(invalid_images))
print("Masks with NaN/Inf:", len(invalid_masks))
print("Masks with unexpected values:", len(invalid_mask_values))

# --------------------------------------------------
# STEP 5: CHECK LABELS
# --------------------------------------------------

print("\nChecking intensity labels...")

missing_wind = labels["wind_speed_knots"].isna().sum()
missing_pressure = labels["pressure_mb"].isna().sum()

print("Missing wind labels:", missing_wind)
print("Missing pressure labels:", missing_pressure)

# --------------------------------------------------
# STEP 6: FINAL SUMMARY
# --------------------------------------------------

print("\n========== FINAL VALIDATION ==========")

checks_passed = (
    len(pair_errors) == 0
    and len(shape_errors) == 0
    and len(invalid_images) == 0
    and len(invalid_masks) == 0
    and len(invalid_mask_values) == 0
    and missing_wind == 0
    and missing_pressure == 0
)

if checks_passed:
    print("SUCCESS: All dataset validation checks passed!")
    print("63 images verified.")
    print("63 masks verified.")
    print("All image-mask pairs are present.")
    print("All image dimensions are correct.")
    print("All mask dimensions are correct.")
    print("All pixel values are valid.")
    print("All intensity labels are available.")
else:
    print("WARNING: Some validation checks failed.")

    if pair_errors:
        print("Missing pairs:", pair_errors[:5])

    if shape_errors:
        print("Shape errors:", shape_errors[:5])

    if invalid_images:
        print("Invalid images:", invalid_images[:5])

    if invalid_masks:
        print("Invalid masks:", invalid_masks[:5])

    if invalid_mask_values:
        print("Unexpected mask values:", invalid_mask_values[:5])

print("======================================")
