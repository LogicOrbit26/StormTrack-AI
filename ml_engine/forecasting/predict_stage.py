
import os
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "weights",
    "stage_temporal_rf.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "weights",
    "stage_temporal_rf_metadata.json"
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_stage_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Stage model not found: {MODEL_PATH}"
        )

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return model, metadata


# Load once when the module is imported
stage_model, stage_metadata = load_stage_model()


# ============================================================
# INPUT VALIDATION
# ============================================================

FEATURES = stage_metadata["features"]

VALID_STAGES = [
    "Developing",
    "Mature",
    "Weakening"
]


def validate_inputs(
    wind_speed_knots,
    pressure_mb,
    wind_change_6h,
    pressure_change_6h
):

    values = {
        "wind_speed_knots": wind_speed_knots,
        "pressure_mb": pressure_mb,
        "wind_change_6h": wind_change_6h,
        "pressure_change_6h": pressure_change_6h
    }

    for name, value in values.items():

        if isinstance(value, bool):
            raise ValueError(
                f"{name} must be a numeric value."
            )

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be numeric."
            )

        if not np.isfinite(number):
            raise ValueError(
                f"{name} must be finite."
            )

    if float(wind_speed_knots) < 0:
        raise ValueError(
            "Wind speed cannot be negative."
        )

    if float(pressure_mb) <= 0:
        raise ValueError(
            "Pressure must be greater than zero."
        )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_stage(
    wind_speed_knots,
    pressure_mb,
    wind_change_6h,
    pressure_change_6h
):

    validate_inputs(
        wind_speed_knots,
        pressure_mb,
        wind_change_6h,
        pressure_change_6h
    )

    input_data = pd.DataFrame(
        [[
            float(wind_speed_knots),
            float(pressure_mb),
            float(wind_change_6h),
            float(pressure_change_6h)
        ]],
        columns=FEATURES
    )

    predicted_stage = stage_model.predict(
        input_data
    )[0]

    probabilities = stage_model.predict_proba(
        input_data
    )[0]

    class_probabilities = {
        str(class_name): round(
            float(probability),
            4
        )
        for class_name, probability in zip(
            stage_model.classes_,
            probabilities
        )
    }

    return {
        "predicted_stage": str(predicted_stage),
        "class_probabilities": class_probabilities,
        "model_name": stage_metadata["model_name"],
        "model_status": "experimental_prototype",
        "label_type": "retrospective_global_peak_labels",
        "warning": (
            "Experimental prediction only. "
            "Not validated for operational cyclone warnings."
        )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CYCLONE LIFECYCLE STAGE PREDICTION")
    print("=" * 60)

    # Example inputs for testing the function only.
    # Replace these with actual observations when integrating.

    result = predict_stage(
        wind_speed_knots=75,
        pressure_mb=970,
        wind_change_6h=-10,
        pressure_change_6h=8
    )

    print("\nPrediction result:")

    print(
        json.dumps(
            result,
            indent=4
        )
    )
