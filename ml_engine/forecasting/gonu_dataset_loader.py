
import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "real_gonu_meteosat7"
)

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")

SPLITS_PATH = os.path.join(DATASET_DIR, "splits.csv")


# --------------------------------------------------
# STEP 1: CREATE SPLIT-AWARE DATASET
# --------------------------------------------------

class GONUDataset(Dataset):

    def __init__(self, split):

        if split not in ["train", "validation", "test"]:
            raise ValueError(
                "Split must be train, validation, or test."
            )

        self.labels = pd.read_csv(SPLITS_PATH)

        self.labels = self.labels[
            self.labels["split"] == split
        ].reset_index(drop=True)

        if len(self.labels) == 0:
            raise ValueError(
                f"No observations found for {split}."
            )

        print(f"\n{split.upper()} dataset initialized!")
        print("Observations:", len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        row = self.labels.iloc[index]

        image_name = row["image_file"]

        mask_name = image_name.replace(
            ".npy",
            "_mask.npy"
        )

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        mask_path = os.path.join(
            MASK_DIR,
            mask_name
        )

        # Load infrared image
        image = np.load(
            image_path,
            allow_pickle=False
        ).astype(np.float32)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Unexpected image shape: {image.shape}"
            )

        # Keep only one infrared channel
        image = image[:, :, 0]

        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or Inf.")

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        # Load validity mask
        mask = np.load(
            mask_path,
            allow_pickle=False
        ).astype(np.float32)

        if mask.ndim != 2:
            raise ValueError(
                f"Unexpected mask shape: {mask.shape}"
            )

        if not np.isfinite(mask).all():
            raise ValueError("Mask contains NaN or Inf.")

        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Unexpected mask values.")

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        # Load target values
        wind = float(row["wind_speed_knots"])
        pressure = float(row["pressure_mb"])

        target = torch.tensor(
            [wind, pressure],
            dtype=torch.float32
        )

        return image, mask, target


# --------------------------------------------------
# STEP 2: TEST ALL THREE DATA SPLITS
# --------------------------------------------------

if __name__ == "__main__":

    print("\nTesting split-aware GONU DataLoader...")

    for split in ["train", "validation", "test"]:

        dataset = GONUDataset(split)

        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=(split == "train"),
            num_workers=0
        )

        images, masks, targets = next(iter(dataloader))

        print("\n========== BATCH INFORMATION ==========")

        print("Split:", split)
        print("Image shape:", images.shape)
        print("Mask shape:", masks.shape)
        print("Target shape:", targets.shape)

        print("Wind range:",
              targets[:, 0].min().item(),
              "to",
              targets[:, 0].max().item())

        print("Pressure range:",
              targets[:, 1].min().item(),
              "to",
              targets[:, 1].max().item())

    print("\nSUCCESS: All three dataset splits loaded!")
