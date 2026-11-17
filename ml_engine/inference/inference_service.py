
import os
import re
import sys
import json
from datetime import datetime, timezone

import numpy as np

# ============================================================
# PATHS AND IMPORTS
# ============================================================

ML_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from forecasting.predict import predict_cyclone_intensity
from forecasting.predict_stage import predict_stage


# ============================================================
# CONFIGURATION
# ============================================================

MAX_INTERVAL_HOURS = 6

MODEL_STATUS = "experimental_prototype"

LABEL_TYPE = "retrospective_global_peak_labels"


# ============================================================
# TIMESTAMP EXTRACTION
# ============================================================

def parse_observation_time(image_path):
    """
    Extract observation timestamp from HURSAT filename.

    Example:
    2007151N14072.GONU.2007.05.31.0600.23.MET-7.013.hursat-b1.v06.npy
    """

    filename = os.path.basename(image_path)

    match = re.search(
        r"\.(\d{4})\.(\d{2})\.(\d{2})\.(\d{4})\.",
        filename
    )

    if not match:
        raise ValueError(
            f"Could not extract timestamp from filename: {filename}"
        )

    year, month, day, hhmm = match.groups()

    hour = int(hhmm[:2])
    minute = int(hhmm[2:])

    if hour > 23 or minute > 59:
        raise ValueError(
            "Invalid observation time in filename."
        )

    return datetime(
        int(year),
        int(month),
        int(day),
        hour,
        minute,
        tzinfo=timezone.utc
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_measurements(wind_speed, pressure):
    """
    Validate wind speed and atmospheric pressure.
    """

    try:
        wind_speed = float(wind_speed)
        pressure = float(pressure)

    except (TypeError, ValueError):
        raise ValueError(
            "Wind speed and pressure must be numeric."
        )

    if not np.isfinite([wind_speed, pressure]).all():
        raise ValueError(
            "Wind speed and pressure must be finite."
        )

    if wind_speed < 0:
        raise ValueError(
            "Wind speed cannot be negative."
        )

    if pressure <= 0:
        raise ValueError(
            "Pressure must be greater than zero."
        )

    return wind_speed, pressure


def validate_image_path(image_path):
    """
    Check whether the satellite image exists.
    """

    if not isinstance(image_path, str):
        raise ValueError(
            "Image path must be a string."
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Satellite image not found: {image_path}"
        )

    if not image_path.lower().endswith(".npy"):
        raise ValueError(
            "Expected a preprocessed .npy satellite image."
        )


# ============================================================
# CNN INTENSITY PREDICTION
# ============================================================

def run_intensity_prediction(image_path):
    """
    Run the trained CNN on one preprocessed satellite image.
    """

    validate_image_path(image_path)

    observation_time = parse_observation_time(
        image_path
    )

    prediction = predict_cyclone_intensity(
        image_path
    )

    wind_speed, pressure = validate_measurements(
        prediction["predicted_wind_speed_knots"],
        prediction["predicted_pressure_mb"]
    )

    return {
        "observation_time": observation_time,
        "wind_speed_knots": wind_speed,
        "pressure_mb": pressure,
        "model_status": prediction["model_status"],
        "image_name": os.path.basename(image_path)
    }


# ============================================================
# TEMPORAL FEATURE CALCULATION
# ============================================================

def calculate_temporal_features(
    previous_observation,
    current_observation
):
    """
    Calculate six-hour-normalized temporal changes.

    Uses the same formula as the Random Forest training script.

    Formula:

        change_6h = difference * 6 / elapsed_hours

    Only intervals greater than zero and at most six hours
    are accepted.
    """

    previous_time = previous_observation[
        "observation_time"
    ]

    current_time = current_observation[
        "observation_time"
    ]

    elapsed_hours = (
        current_time - previous_time
    ).total_seconds() / 3600

    if elapsed_hours <= 0:
        raise ValueError(
            "Current observation must be later than "
            "the previous observation."
        )

    if elapsed_hours > MAX_INTERVAL_HOURS:
        raise ValueError(
            f"Observations are {elapsed_hours:.2f} hours apart. "
            "Maximum allowed interval is 6 hours."
        )

    previous_wind = previous_observation[
        "wind_speed_knots"
    ]

    current_wind = current_observation[
        "wind_speed_knots"
    ]

    previous_pressure = previous_observation[
        "pressure_mb"
    ]

    current_pressure = current_observation[
        "pressure_mb"
    ]

    wind_difference = (
        current_wind - previous_wind
    )

    pressure_difference = (
        current_pressure - previous_pressure
    )

    wind_change_6h = (
        wind_difference * 6 / elapsed_hours
    )

    pressure_change_6h = (
        pressure_difference * 6 / elapsed_hours
    )

    return {
        "elapsed_hours": round(
            elapsed_hours,
            3
        ),

        "wind_change_6h": round(
            wind_change_6h,
            4
        ),

        "pressure_change_6h": round(
            pressure_change_6h,
            4
        )
    }


# ============================================================
# LIFECYCLE STAGE PREDICTION
# ============================================================

def run_stage_prediction(
    current_observation,
    temporal_features
):
    """
    Run the temporal Random Forest stage classifier.

    Inputs are CNN-generated wind and pressure estimates.
    This is an experimental combination because the RF was
    trained using IBTrACS observations.
    """

    result = predict_stage(
        wind_speed_knots=current_observation[
            "wind_speed_knots"
        ],

        pressure_mb=current_observation[
            "pressure_mb"
        ],

        wind_change_6h=temporal_features[
            "wind_change_6h"
        ],

        pressure_change_6h=temporal_features[
            "pressure_change_6h"
        ]
    )

    return {
        "predicted_stage": result[
            "predicted_stage"
        ],

        "class_probabilities": result[
            "class_probabilities"
        ],

        "model_name": result[
            "model_name"
        ],

        "model_status": result[
            "model_status"
        ],

        "label_type": result[
            "label_type"
        ],

        "stage_input_source": "cnn_prediction",

        "temporal_features": temporal_features
    }


# ============================================================
# COMBINED INFERENCE SERVICE
# ============================================================

def run_inference(
    current_image_path,
    previous_image_path=None
):
    """
    Complete cyclone inference pipeline.

    Step 1: Predict current wind and pressure.
    Step 2: If a previous image is available, predict its
            wind and pressure.
    Step 3: Calculate temporal changes.
    Step 4: Predict lifecycle stage.

    Stage prediction is unavailable if no valid previous
    observation exists.
    """

    print("\nRunning cyclone inference...")

    # --------------------------------------------------------
    # CURRENT OBSERVATION
    # --------------------------------------------------------

    current_observation = run_intensity_prediction(
        current_image_path
    )

    current_time = current_observation[
        "observation_time"
    ]

    print(
        "Current observation:",
        current_time.isoformat()
    )

    print(
        "Current CNN wind:",
        current_observation["wind_speed_knots"]
    )

    print(
        "Current CNN pressure:",
        current_observation["pressure_mb"]
    )

    # --------------------------------------------------------
    # BASE RESPONSE
    # --------------------------------------------------------

    response = {
        "observation_time": current_time.isoformat(),

        "intensity": {
            "predicted_wind_speed_knots":
                current_observation["wind_speed_knots"],

            "predicted_pressure_mb":
                current_observation["pressure_mb"],

            "model_status": MODEL_STATUS,

            "input_image":
                current_observation["image_name"]
        },

        "stage": {
            "predicted_stage": None,
            "class_probabilities": None,
            "model_name": "Temporal Random Forest",
            "model_status": MODEL_STATUS,
            "label_type": LABEL_TYPE,
            "stage_input_source": "cnn_prediction",
            "temporal_features": None,
            "unavailable_reason": None
        },

        "warnings": [
            "Experimental research prototype only.",

            "CNN intensity predictions have limited "
            "cross-storm generalization.",

            "Random Forest was trained on IBTrACS observations, "
            "not CNN-generated measurements.",

            "Lifecycle labels are retrospective global-peak labels.",

            "Not validated for operational cyclone warnings."
        ]
    }

    # --------------------------------------------------------
    # PREVIOUS OBSERVATION CHECK
    # --------------------------------------------------------

    if previous_image_path is None:

        response["stage"]["unavailable_reason"] = (
            "Previous satellite observation is required "
            "to calculate temporal features."
        )

        return response

    # --------------------------------------------------------
    # PREVIOUS CNN PREDICTION
    # --------------------------------------------------------

    previous_observation = run_intensity_prediction(
        previous_image_path
    )

    previous_time = previous_observation[
        "observation_time"
    ]

    print(
        "\nPrevious observation:",
        previous_time.isoformat()
    )

    print(
        "Previous CNN wind:",
        previous_observation["wind_speed_knots"]
    )

    print(
        "Previous CNN pressure:",
        previous_observation["pressure_mb"]
    )

    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    try:

        temporal_features = calculate_temporal_features(
            previous_observation,
            current_observation
        )

    except ValueError as exc:

        response["stage"]["unavailable_reason"] = str(exc)

        return response

    print("\nTemporal features:")

    print(
        json.dumps(
            temporal_features,
            indent=4
        )
    )

    # --------------------------------------------------------
    # STAGE PREDICTION
    # --------------------------------------------------------

    stage_result = run_stage_prediction(
        current_observation,
        temporal_features
    )

    response["stage"].update(
        stage_result
    )

    response["stage"]["unavailable_reason"] = None

    return response


# ============================================================
# DATASET HELPERS
# ============================================================

def get_sorted_images(image_directory):
    """
    Return chronologically sorted preprocessed .npy images.
    """

    if not os.path.isdir(image_directory):
        raise FileNotFoundError(
            f"Image directory not found: {image_directory}"
        )

    image_paths = []

    for filename in os.listdir(image_directory):

        if not filename.lower().endswith(".npy"):
            continue

        full_path = os.path.join(
            image_directory,
            filename
        )

        try:
            observation_time = parse_observation_time(
                full_path
            )

            image_paths.append(
                (observation_time, full_path)
            )

        except ValueError:
            continue

    image_paths.sort(
        key=lambda item: item[0]
    )

    return [
        path for _, path in image_paths
    ]


# ============================================================
# EXPORT JSON AND SATELLITE PNG
# ============================================================

def create_satellite_heatmap(image_path, output_path):
    from PIL import Image

    image = np.load(image_path)

    if image.ndim == 3:
        image = image[..., 0]

    image = np.nan_to_num(
        image.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    low, high = np.percentile(image, [2, 98])

    if high > low:
        image = np.clip((image - low) / (high - low), 0, 1)
    else:
        image = np.zeros_like(image)

    grayscale = (image * 255).astype(np.uint8)

    # False-colour visualization, not an SST prediction
    rgb = np.stack([
        grayscale,
        np.roll(grayscale, 60, axis=0),
        255 - grayscale
    ], axis=-1)

    Image.fromarray(rgb).save(output_path)


def export_inference_outputs(result, image_path):
    output_dir = os.path.join(ML_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "predictions.json")
    png_path = os.path.join(output_dir, "satellite_heatmap.png")

    create_satellite_heatmap(image_path, png_path)

    export_data = dict(result)

    export_data["visualization"] = {
        "satellite_heatmap": "satellite_heatmap.png",
        "source_image": os.path.basename(image_path)
    }

    export_data["coordinates"] = None
    export_data["coordinate_note"] = (
        "Coordinates are not predicted by the current ML models."
    )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(export_data, file, indent=4, allow_nan=False)

    print("\nExported JSON:", json_path)
    print("Exported PNG:", png_path)

    return json_path, png_path

# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    SAMPLE_DIR = os.path.join(
        ML_DIR,
        "dataset",
        "processed",
        "real_gonu_meteosat7",
        "images"
    )

    image_paths = get_sorted_images(
        SAMPLE_DIR
    )

    if len(image_paths) < 2:
        raise FileNotFoundError(
            "At least two valid satellite images are required."
        )

    previous_image = image_paths[0]
    current_image = image_paths[1]

    print("=" * 65)
    print("STORMTRACK-AI COMBINED INFERENCE")
    print("=" * 65)

    print("\nPrevious image:")
    print(os.path.basename(previous_image))

    print("\nCurrent image:")
    print(os.path.basename(current_image))

    result = run_inference(
        current_image_path=current_image,
        previous_image_path=previous_image
    )
    export_inference_outputs(result, current_image)
    print("\n" + "=" * 65)
    print("FINAL INFERENCE RESULT")
    print("=" * 65)

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print("\n" + "=" * 65)
    print("INFERENCE COMPLETED")
    print("=" * 65)
