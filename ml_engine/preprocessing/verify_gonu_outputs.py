
import os
import csv
import numpy as np

BASE = r"C:\StormTrack-AI\ml_engine\dataset\processed\real_gonu"

IMAGE_DIR = os.path.join(BASE, "images")
MASK_DIR = os.path.join(BASE, "masks")
CSV_PATH = os.path.join(BASE, "metadata.csv")

images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".npy")]
masks = [f for f in os.listdir(MASK_DIR) if f.endswith("_mask.npy")]

print("NPY images:", len(images))
print("Masks:", len(masks))

# Inspect one image and its corresponding mask
image_name = images[0]
base_name = image_name[:-4]

image = np.load(os.path.join(IMAGE_DIR, image_name))
mask = np.load(os.path.join(MASK_DIR, base_name + "_mask.npy"))

print("\nSample image:", image_name)
print("Image shape:", image.shape)
print("Image minimum:", image.min())
print("Image maximum:", image.max())

print("\nMask shape:", mask.shape)
print("Mask values:", np.unique(mask))
print("Valid pixels:", np.sum(mask == 1))
print("Invalid pixels:", np.sum(mask == 0))

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("\nMetadata rows:", len(rows))
print("First metadata record:")
print(rows[0])
