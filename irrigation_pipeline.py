"""
Terrace Irrigation ML Pipeline
==============================

Extracted and adapted from:
terrace_irrigation_pipeline.ipynb

Pipeline:
1. Calibrate the raw soil moisture ADC reading.
2. Estimate crop evapotranspiration demand.
3. Predict future soil moisture using Random Forest.
4. Apply the rainfall discount.
5. Return an irrigation-need score and recommendation.

IMPORTANT:
The current Random Forest is trained on synthetic data.
Its predictions are for software integration and demonstration,
not yet scientifically validated irrigation recommendations.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_VERSION = "terrace-rf-synthetic-v1"

# Sensor calibration values from the original notebook.
ADC_AIR_DRY = 1880.0
ADC_WATER_WET = 1200.0

# Ginger crop parameter from the original notebook.
GINGER_KC = 1.10

# Irrigation scoring parameters from the original notebook.
FIELD_CAPACITY = 0.55
STRESS_THRESHOLD = 0.34  # Retained from notebook; not used directly.

# Keep this order identical to the model's training feature order.
FEATURE_NAMES = [
    "initial_moisture",
    "estimated_etc",
    "terrain_index",
    "temperature_c",
    "forecast_hours",
]


# ============================================================
# 2. SOIL MOISTURE SENSOR CALIBRATION
# ============================================================

def calibrate_moisture_adc(raw_adc: float) -> float:
    """
    Convert a raw sensor ADC reading into a normalized
    moisture index between 0.0 (dry) and 1.0 (wet).

    Calibration taken from the original notebook:
        ADC 1880 -> 0.0
        ADC 1200 -> 1.0

    These values should eventually be calibrated against
    the actual sensor and soil used in the experiment.
    """

    raw_adc = float(raw_adc)

    if not np.isfinite(raw_adc):
        raise ValueError("raw_adc must be a finite number.")

    if not 0 <= raw_adc <= 4095:
        raise ValueError("raw_adc must be between 0 and 4095.")

    moisture = (
        (ADC_AIR_DRY - raw_adc)
        / (ADC_AIR_DRY - ADC_WATER_WET)
    )

    return float(np.clip(moisture, 0.0, 1.0))


# ============================================================
# 3. CROP WATER-DEMAND ESTIMATION
# ============================================================

def estimate_crop_etc(
    temperature_c: float,
    hour_of_day: float = 13.0,
    rh_percent: float = 70.0,
) -> float:
    """
    Estimate ginger crop water demand using the simplified
    calculation in the original notebook.

    Returns an estimated ETc-like value in mm/hour.

    NOTE:
    This preserves the notebook's approximation. It is not a
    complete implementation of the FAO-56 Penman-Monteith
    equation because required meteorological inputs such as
    wind speed and a full radiation calculation are absent.
    """

    temperature_c = float(temperature_c)
    hour_of_day = float(hour_of_day)
    rh_percent = float(rh_percent)

    if not np.isfinite(temperature_c):
        raise ValueError("temperature_c must be finite.")

    if not np.isfinite(hour_of_day):
        raise ValueError("hour_of_day must be finite.")

    if not 0.0 <= rh_percent <= 100.0:
        raise ValueError("rh_percent must be between 0 and 100.")

    # Original notebook's approximate daytime solar radiation.
    if 7.0 <= hour_of_day <= 19.0:
        solar_rad_w = 750.0 * np.sin(
            np.pi * (hour_of_day - 7.0) / 12.0
        )
    else:
        solar_rad_w = 0.0

    # Approximate radiation conversion used in the notebook.
    rn_mm_hr = (solar_rad_w * 0.0036) / 2.45

    # Saturation and actual vapour pressure.
    sat_vp = 0.6108 * np.exp(
        (17.27 * temperature_c) / (temperature_c + 237.3)
    )

    act_vp = sat_vp * (rh_percent / 100.0)

    vapour_pressure_deficit = max(
        0.0,
        sat_vp - act_vp,
    )

    # Simplified reference evapotranspiration estimate.
    et0_mm_hr = max(
        0.02,
        0.408 * rn_mm_hr * 0.65
        + 0.05 * vapour_pressure_deficit,
    )

    # Ginger crop coefficient from the notebook.
    etc_mm_hr = GINGER_KC * et0_mm_hr

    return float(etc_mm_hr)


# ============================================================
# 4. RANDOM FOREST MODEL TRAINING
# ============================================================

def train_rf_moisture_decay_model():
    """
    Train the Random Forest moisture-decay model using the
    synthetic data-generation procedure from the notebook.

    Training features:
        1. Initial normalized moisture
        2. Estimated crop water demand
        3. Normalized terrain index
        4. Temperature
        5. Forecast horizon

    The target is predicted future normalized moisture.

    IMPORTANT:
    These are synthetic examples, not measurements of actual
    soil moisture over time.
    """

    # RandomState preserves the notebook's legacy random-number
    # generation method and seed.
    rng = np.random.RandomState(42)

    n_samples = 1500

    initial_moisture = rng.uniform(
        0.10, 0.85, n_samples
    )

    etc_values = rng.uniform(
        0.05, 0.60, n_samples
    )

    terrain_indices = rng.uniform(
        0.0, 1.0, n_samples
    )

    temperatures = rng.uniform(
        15.0, 32.0, n_samples
    )

    forecast_hours = rng.uniform(
        6.0, 8.0, n_samples
    )

    # Moisture depletion assumptions from the notebook.
    drainage_rate = (
        0.012 * (1.0 + 0.5 * terrain_indices)
    )

    evaporation_loss = (
        etc_values * 0.035
    ) * (temperatures / 25.0)

    total_decay = (
        drainage_rate + evaporation_loss
    ) * forecast_hours

    noise = rng.normal(
        0.0, 0.008, n_samples
    )

    future_moisture = np.clip(
        initial_moisture - total_decay + noise,
        0.04,
        0.90,
    )

    # Preserve the original training feature order.
    X = np.column_stack([
        initial_moisture,
        etc_values,
        terrain_indices,
        temperatures,
        forecast_hours,
    ])

    y = future_moisture

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        random_state=42,
    )

    model.fit(X, y)

    return model


# Train once when this module is imported.
# The API should reuse this model for subsequent requests.
ML_MODEL = train_rf_moisture_decay_model()


# ============================================================
# 5. IRRIGATION SCORE AND RECOMMENDATION
# ============================================================

def calculate_irrigation_score(
    predicted_moisture: float,
    rain_probability_8h: float,
) -> float:
    """
    Calculate the irrigation-need score from 0.0 to 1.0.

    Preserves the original notebook's equations:

        deficit_factor =
            (field_capacity - predicted_moisture)
            / (field_capacity - 0.15)

        rain_discount =
            1 - (rain_probability * 1.5)

        irrigation_score =
            deficit_factor * rain_discount

    The result is clipped to [0.0, 1.0].
    """

    predicted_moisture = float(predicted_moisture)
    rain_probability_8h = float(rain_probability_8h)

    if not np.isfinite(predicted_moisture):
        raise ValueError("predicted_moisture must be finite.")

    if not 0.0 <= rain_probability_8h <= 1.0:
        raise ValueError(
            "rain_probability_8h must be between 0 and 1."
        )

    deficit_factor = max(
        0.0,
        (FIELD_CAPACITY - predicted_moisture)
        / (FIELD_CAPACITY - 0.15),
    )

    rain_discount = max(
        0.0,
        1.0 - (rain_probability_8h * 1.5),
    )

    irrigation_score = np.clip(
        deficit_factor * rain_discount,
        0.0,
        1.0,
    )

    return float(irrigation_score)


def get_irrigation_recommendation(
    irrigation_need_score: float,
) -> str:
    """
    Convert the score into the recommendation categories
    used by the FastAPI service.
    """

    if irrigation_need_score >= 0.65:
        return "HIGH"

    if irrigation_need_score >= 0.35:
        return "MODERATE"

    return "STANDBY"


# ============================================================
# 6. COMPLETE IRRIGATION PREDICTION PIPELINE
# ============================================================

def predict_irrigation(
    raw_adc: float,
    temperature_c: float,
    terrain_index: float,
    forecast_hours: float = 7.0,
    rain_probability_8h: float = 0.15,
) -> dict:
    """
    Run the complete single-sensor irrigation prediction.

    Parameters
    ----------
    raw_adc:
        Raw soil moisture ADC reading from the Arduino.

    temperature_c:
        Temperature in degrees Celsius.

    terrain_index:
        Normalized terrain index between 0.0 and 1.0.
        This must be supplied by the caller.

    forecast_hours:
        Prediction horizon. The model was trained on horizons
        between 6 and 8 hours.

    rain_probability_8h:
        Probability of rain during the coming eight hours,
        expressed as a fraction between 0.0 and 1.0.

    Returns
    -------
    Dictionary containing current moisture, predicted
    moisture, estimated water demand, irrigation score,
    recommendation, and model version.
    """

    # Validate environmental inputs.
    temperature_c = float(temperature_c)
    terrain_index = float(terrain_index)
    forecast_hours = float(forecast_hours)
    rain_probability_8h = float(rain_probability_8h)

    numeric_inputs = [
        temperature_c,
        terrain_index,
        forecast_hours,
        rain_probability_8h,
    ]

    if not all(np.isfinite(value) for value in numeric_inputs):
        raise ValueError("All pipeline inputs must be finite.")

    if not 0.0 <= terrain_index <= 1.0:
        raise ValueError(
            "terrain_index must be between 0 and 1."
        )

    if not 6.0 <= forecast_hours <= 8.0:
        raise ValueError(
            "forecast_hours must be between 6 and 8 "
            "for this model."
        )

    if not 0.0 <= rain_probability_8h <= 1.0:
        raise ValueError(
            "rain_probability_8h must be between 0 and 1."
        )

    # Step 1: Convert the sensor reading to normalized moisture.
    current_moisture = calibrate_moisture_adc(raw_adc)

    # Step 2: Get the current hour in India Standard Time.
    india_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    current_hour = (
        india_time.hour + india_time.minute / 60.0
    )

    # Step 3: Estimate crop water demand.
    estimated_etc = estimate_crop_etc(
        temperature_c=temperature_c,
        hour_of_day=current_hour,
        rh_percent=70.0,
    )

    # Step 4: Build the model input in exactly the same order
    # used during training.
    model_input = np.array([[
        current_moisture,
        estimated_etc,
        terrain_index,
        temperature_c,
        forecast_hours,
    ]])

    # Step 5: Predict future moisture.
    predicted_moisture = float(
        ML_MODEL.predict(model_input)[0]
    )

    # Step 6: Compute the irrigation-need score.
    irrigation_need_score = calculate_irrigation_score(
        predicted_moisture=predicted_moisture,
        rain_probability_8h=rain_probability_8h,
    )

    # Step 7: Generate a recommendation.
    recommendation = get_irrigation_recommendation(
        irrigation_need_score
    )

    # Step 8: Return the API-friendly result.
    return {
        "current_moisture": round(
            current_moisture, 3
        ),

        "predicted_moisture": round(
            predicted_moisture, 3
        ),

        "estimated_etc": round(
            estimated_etc, 3
        ),

        "rain_probability_8h": round(
            rain_probability_8h, 3
        ),

        "irrigation_need_score": round(
            irrigation_need_score, 3
        ),

        "recommendation": recommendation,

        "model_version": MODEL_VERSION,
    }