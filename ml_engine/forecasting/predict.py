import io
import os
import sys
import threading

import torch
import numpy as np

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ML_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, ML_DIR)

from models.cyclone_cnn import CycloneIntensityModel

WEIGHTS_PATH = os.path.join(
    ML_DIR,
    "weights",
    "gonu_cyclone_resnet18.pth"
)

DEVICE = torch.device("cpu")

# Cache the model so it loads only once
_MODEL_CACHE = None
_MODEL_LOCK = threading.Lock()


# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------

def load_inference_model():

    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    with _MODEL_LOCK:

        # Check again after acquiring the lock
        if _MODEL_CACHE is not None:
            return _MODEL_CACHE

        if not os.path.isfile(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found: {WEIGHTS_PATH}"
            )

        checkpoint = torch.load(
            WEIGHTS_PATH,
            map_location=DEVICE
        )

        model = CycloneIntensityModel(
            pretrained=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(DEVICE)
        model.eval()

        target_mean = checkpoint["target_mean"].to(DEVICE)
        target_std = checkpoint["target_std"].to(DEVICE)

        _MODEL_CACHE = (
            model,
            target_mean,
            target_std
        )

        print("GONU CNN model loaded successfully!")

    return _MODEL_CACHE


# --------------------------------------------------
# PREDICT CYCLONE INTENSITY
# --------------------------------------------------

def predict_cyclone_intensity(image_path):

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Reuse cached model
    model, target_mean, target_std = load_inference_model()

    # Load preprocessed satellite image
    image = np.load(
        image_path,
        allow_pickle=False
    ).astype(np.float32)

    if image.shape != (301, 301, 3):
        raise ValueError(
            f"Expected image shape (301, 301, 3), got {image.shape}"
        )

    # Use first infrared channel
    image = image[:, :, 0]

    if not np.isfinite(image).all():
        raise ValueError(
            "Image contains invalid values."
        )

    # Convert to tensor [1, 1, 301, 301]
    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # Predict
    with torch.inference_mode():

        normalized_prediction = model(image_tensor)

        prediction = (
            normalized_prediction * target_std + target_mean
        )

    wind_speed = prediction[0, 0].item()
    pressure = prediction[0, 1].item()

    if not np.isfinite([wind_speed, pressure]).all():
        raise ValueError(
            "Model returned invalid predictions."
        )

    return {
        "predicted_wind_speed_knots": round(wind_speed, 2),
        "predicted_pressure_mb": round(pressure, 2),
        "model_status": "experimental_prototype"
    }

def predict_from_array(image):

    model, target_mean, target_std = load_inference_model()

    image = np.asarray(image, dtype=np.float32)

    if image.shape != (301, 301, 3):
        raise ValueError(
            f"Expected image shape (301, 301, 3), got {image.shape}"
        )

    image = image[:, :, 0]

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid values.")

    image_tensor = torch.from_numpy(
        image.copy()
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        normalized_prediction = model(image_tensor)
        prediction = normalized_prediction * target_std + target_mean

    wind_speed = prediction[0, 0].item()
    pressure = prediction[0, 1].item()

    if not np.isfinite([wind_speed, pressure]).all():
        raise ValueError("Model returned invalid predictions.")

    return {
        "predicted_wind_speed_knots": round(wind_speed, 2),
        "predicted_pressure_mb": round(pressure, 2),
        "model_status": "experimental_prototype"
    }


def predict_from_bytes(image_bytes):

    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    try:
        image = np.load(
            io.BytesIO(image_bytes),
            allow_pickle=False
        )
    except Exception as exc:
        raise ValueError(
            "Invalid upload. Expected a preprocessed .npy file."
        ) from exc

    return predict_from_array(image)

# TEST
# --------------------------------------------------

if __name__ == "__main__":

    sample_dir = os.path.join(
        ML_DIR,
        "dataset",
        "processed",
        "real_gonu_meteosat7",
        "images"
    )

    sample_files = sorted([
        file for file in os.listdir(sample_dir)
        if file.endswith(".npy")
    ])

    if not sample_files:
        raise FileNotFoundError(
            "No preprocessed satellite images found."
        )

    sample_path = os.path.join(
        sample_dir,
        sample_files[0]
    )

    print("\nTesting GONU CNN inference...")

    result = predict_cyclone_intensity(sample_path)

    print("\n========== PREDICTION RESULT ==========")
    print("Wind speed:", result["predicted_wind_speed_knots"], "knots")
    print("Pressure:", result["predicted_pressure_mb"], "mb")
    print("Status:", result["model_status"])

    print("\nInference test completed!")
