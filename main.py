"""Terrace Irrigation API with local cardboard-model GIS features.

GIS features are returned for verification only; they do not alter the
existing Random Forest prediction or Arduino pump safety logic.
"""

import logging
import math
import os
import secrets
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from irrigation_pipeline import MODEL_VERSION, predict_irrigation
from gis_features import (
    GISFeatureError,
    LocationOutsideRasterError,
    NoTerrainDataError,
    get_raster_info,
    get_terrain_sample,
    get_terrain_sample_for_zone,
    get_terrain_feature_ranges,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Terrace Irrigation API",
    description=(
        "ML-assisted irrigation predictions using soil moisture, temperature, "
        "terrain and rainfall context. GIS terrain features are extracted from "
        "a local cardboard terrain model using X/Y coordinates in centimetres. "
        "GIS does not affect the Random Forest prediction; a separate experimental GIS-aware score is returned only when local coordinates and real-soil calibration are configured."
    ),
    version="1.2.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Depends(api_key_header)) -> None:
    """Protect API endpoints using the configured X-API-Key header."""
    expected_key = os.getenv("IRRIGATION_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )
    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ============================= SENSOR LOCAL-MODEL LOOKUP =============================

def get_gis_payload_for_sensor(sensor_id: str) -> dict[str, Any]:
    """Read local board X/Y configuration for the given sensor.

    Set TERRACE_SENSOR_X_CM and TERRACE_SENSOR_Y_CM in Render after the physical
    sensor/monitored area has been measured. Until then, GIS is marked as not
    configured and the original prediction still succeeds.
    """

    def unavailable(status: str, message: str) -> dict[str, Any]:
        return {
            "gis_status": status,
            "gis_message": message,
            "gis_model_location": None,
            "gis_raster": None,
            "terrain_features": None,
        }

    if sensor_id != "terrace_sensor_01":
        return unavailable(
            "location_not_configured",
            "No local model position is configured for this sensor ID.",
        )

    x_text = os.getenv("TERRACE_SENSOR_X_CM")
    y_text = os.getenv("TERRACE_SENSOR_Y_CM")
    if not x_text and not y_text:
        return unavailable(
            "location_not_configured",
            "Set TERRACE_SENSOR_X_CM and TERRACE_SENSOR_Y_CM after measuring the position on the cardboard model.",
        )
    if not x_text or not y_text:
        return unavailable(
            "configuration_error",
            "Both TERRACE_SENSOR_X_CM and TERRACE_SENSOR_Y_CM must be configured.",
        )

    try:
        sample = get_terrain_sample(float(x_text), float(y_text))
        return {
            "gis_status": "ok",
            "gis_message": None,
            "gis_model_location": sample["model_location"],
            "gis_raster": sample["raster"],
            "terrain_features": sample["features"],
        }
    except LocationOutsideRasterError as exc:
        return unavailable("outside_raster", str(exc))
    except NoTerrainDataError as exc:
        return unavailable("no_data", str(exc))
    except (ValueError, TypeError) as exc:
        return unavailable("configuration_error", str(exc))
    except GISFeatureError:
        logger.exception("GIS extraction failed for sensor %s", sensor_id)
        return unavailable(
            "unavailable",
            "GIS data could not be read. The existing prediction was not changed.",
        )
    except Exception:
        logger.exception("Unexpected GIS error for sensor %s", sensor_id)
        return unavailable(
            "error",
            "An unexpected GIS error occurred. The existing prediction was not changed.",
        )



# ============================= EXPERIMENTAL GIS-AWARE SCORE =============================

def calculate_experimental_gis_aware_score(
    raw_adc: int,
    estimated_etc: float,
    rain_probability_8h: float,
    gis_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return a separate provisional score; never replace the existing RF score.

    The score is intentionally unavailable until the sensor location and real-soil
    dry/wet ADC calibration are configured. Its heuristic weights are provisional
    and must not be used for automatic pump control.
    """
    unavailable = {
        "experimental_gis_aware_score": None,
        "experimental_score_status": "not_available",
        "experimental_score_components": None,
        "gis_used_for_experimental_score": False,
    }

    if gis_payload.get("gis_status") != "ok":
        unavailable["experimental_score_status"] = (
            "gis_" + str(gis_payload.get("gis_status", "unavailable"))
        )
        return unavailable

    dry_text = os.getenv("SOIL_ADC_DRY")
    wet_text = os.getenv("SOIL_ADC_WET")
    if not dry_text or not wet_text:
        unavailable["experimental_score_status"] = "soil_calibration_not_configured"
        return unavailable

    try:
        adc_dry = float(dry_text)
        adc_wet = float(wet_text)
    except (TypeError, ValueError):
        unavailable["experimental_score_status"] = "soil_calibration_invalid"
        return unavailable

    if (
        not (0.0 <= adc_wet <= 4095.0)
        or not (0.0 <= adc_dry <= 4095.0)
        or adc_dry <= adc_wet
    ):
        unavailable["experimental_score_status"] = "soil_calibration_invalid"
        return unavailable

    features = gis_payload.get("terrain_features") or {}
    try:
        slope = float(features["slope_deg"])
        twi = float(features["twi"])
        if not (math.isfinite(slope) and math.isfinite(twi)):
            raise ValueError("Slope and TWI must be finite.")
        ranges = get_terrain_feature_ranges()
    except (KeyError, TypeError, ValueError):
        unavailable["experimental_score_status"] = "terrain_features_invalid"
        return unavailable
    except GISFeatureError:
        logger.exception("Could not read terrain feature ranges for score.")
        unavailable["experimental_score_status"] = "terrain_ranges_unavailable"
        return unavailable

    def normalize(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    # Higher ADC means drier for the current sensor orientation.
    calibrated_moisture = max(
        0.0, min(1.0, (adc_dry - float(raw_adc)) / (adc_dry - adc_wet))
    )
    moisture_deficit = 1.0 - calibrated_moisture

    etc_reference_max = 0.60  # Upper end of the existing synthetic training range.
    etc_normalized = max(0.0, min(1.0, float(estimated_etc) / etc_reference_max))

    slope_norm = normalize(slope, ranges["slope_deg"]["min"], ranges["slope_deg"]["max"])
    twi_norm = normalize(twi, ranges["twi"]["min"], ranges["twi"]["max"])

    # Explicitly provisional weights: slope may increase runoff-related need;
    # greater TWI indicates higher relative wetness potential and reduces need.
    # These weights are engineering assumptions, not fitted/validated parameters.
    moisture_weight = 0.80
    evapotranspiration_weight = 0.20
    slope_weight = 0.10
    twi_weight = 0.10

    base_need = (
        moisture_weight * moisture_deficit
        + evapotranspiration_weight * etc_normalized
    )
    terrain_multiplier = max(
        0.80, min(1.20, 1.0 + slope_weight * slope_norm - twi_weight * twi_norm)
    )
    rain_discount = max(0.0, 1.0 - 1.5 * float(rain_probability_8h))
    score = max(0.0, min(1.0, base_need * terrain_multiplier * rain_discount))

    return {
        "experimental_gis_aware_score": round(score, 4),
        "experimental_score_status": "available_experimental_not_validated",
        "experimental_score_components": {
            "calibrated_moisture_estimate": round(calibrated_moisture, 4),
            "moisture_deficit": round(moisture_deficit, 4),
            "estimated_etc_normalized": round(etc_normalized, 4),
            "slope_normalized_to_this_raster": round(slope_norm, 4),
            "twi_normalized_to_this_raster": round(twi_norm, 4),
            "terrain_multiplier": round(terrain_multiplier, 4),
            "rain_discount": round(rain_discount, 4),
        },
        "gis_used_for_experimental_score": True,
    }

# =================================== REQUEST SCHEMA ===================================

class PredictionRequest(BaseModel):
    """Incoming data expected from the Arduino."""

    sensor_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Unique identifier for the soil sensor.",
    )
    raw_adc: int = Field(ge=0, le=4095, description="Actual raw sensor ADC reading.")
    temperature_c: float = Field(
        ge=-10, le=60, allow_inf_nan=False, description="Temperature in degrees Celsius."
    )
    terrain_index: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="Legacy normalized terrain model input; not GIS TWI.",
    )
    forecast_hours: float = Field(
        ge=6, le=8, allow_inf_nan=False, description="Prediction horizon in hours."
    )
    rain_probability_8h: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="Rain probability over eight hours, as a fraction.",
    )


# =================================== RESPONSE SCHEMA ==================================

class GISModelLocation(BaseModel):
    x_cm: float
    y_cm: float
    origin: str
    x_direction: str
    y_direction: str


class GISRasterDetails(BaseModel):
    source: str
    crs_metadata: str
    pixel_row: int
    pixel_col: int
    pixel_center_x_cm: float
    pixel_center_y_cm: float
    board_width_cm: float
    board_height_cm: float
    cell_width_cm: float
    cell_height_cm: float


class TerrainFeatures(BaseModel):
    elevation_m: float
    slope_deg: float
    aspect_deg: float | None = None
    curvature: float
    tpi_m: float
    twi: float


class PredictionResponse(BaseModel):
    """Original prediction plus sensor input and local GIS diagnostics."""

    sensor_id: str
    raw_adc: int
    temperature_c: float
    terrain_index: float
    forecast_hours: float
    current_moisture: float
    predicted_moisture: float
    estimated_etc: float
    rain_probability_8h: float
    irrigation_need_score: float
    recommendation: str
    model_version: str
    gis_status: str = "location_not_configured"
    gis_message: str | None = None
    gis_model_location: GISModelLocation | None = None
    gis_raster: GISRasterDetails | None = None
    terrain_features: TerrainFeatures | None = None
    gis_used_for_prediction: bool = False
    model_validation_status: str = (
        "synthetic_training_data_not_validated_on_real_measurements"
    )
    experimental_gis_aware_score: float | None = None
    experimental_score_status: str = "requires_gis_location_and_sensor_calibration"
    experimental_score_components: dict[str, float] | None = None
    experimental_score_formula_version: str = "experimental-rule-based-v1"
    gis_used_for_experimental_score: bool = False
    experimental_score_warning: str = (
        "Experimental heuristic only; not independently validated and must not control the pump."
    )


# ====================================== ENDPOINTS =====================================

@app.get("/health")
def health_check() -> dict[str, str]:
    """Public health endpoint for the hosting service."""
    return {
        "status": "ok",
        "service": "Terrace Irrigation API",
        "model_version": MODEL_VERSION,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Terrace Irrigation API",
        "status": "running",
        "health_endpoint": "/health",
        "documentation": "/docs",
        "prediction_endpoint": "/predict",
        "gis_info_endpoint": "/gis/info",
        "gis_test_point_endpoint": "/gis/test-point",
        "gis_test_zone_endpoint": "/gis/test-zone/{zone_code}",
    }


@app.get("/gis/info")
def gis_info(api_key: str | None = Depends(api_key_header)) -> dict[str, Any]:
    """Return the raster and local cardboard-model extent metadata."""
    require_api_key(api_key)
    try:
        return get_raster_info()
    except GISFeatureError as exc:
        logger.exception("Could not read GIS raster metadata.")
        raise HTTPException(status_code=503, detail="The GIS raster is unavailable.") from exc


@app.get("/gis/test-point")
def gis_test_point(
    x_cm: float = Query(..., description="X coordinate in centimetres from the model's left edge."),
    y_cm: float = Query(..., description="Y coordinate in centimetres from the model's top edge."),
    api_key: str | None = Depends(api_key_header),
) -> dict[str, Any]:
    """Test a local model point before the physical sensor position is measured."""
    require_api_key(api_key)
    try:
        return get_terrain_sample(x_cm=x_cm, y_cm=y_cm)
    except LocationOutsideRasterError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GISFeatureError as exc:
        logger.exception("Could not extract terrain at local model point.")
        raise HTTPException(status_code=503, detail="Terrain extraction failed.") from exc


@app.get("/gis/test-zone/{zone_code}")
def gis_test_zone(
    zone_code: str,
    api_key: str | None = Depends(api_key_header),
) -> dict[str, Any]:
    """Test a pre-labelled sample zone; this does not identify the sensor position."""
    require_api_key(api_key)
    try:
        return get_terrain_sample_for_zone(zone_code.upper())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GISFeatureError as exc:
        logger.exception("GIS extraction failed for zone %s", zone_code)
        raise HTTPException(status_code=503, detail="Terrain extraction failed.") from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    api_key: str | None = Depends(api_key_header),
) -> PredictionResponse:
    """Run existing prediction and append GIS information if configured."""
    require_api_key(api_key)

    try:
        result = predict_irrigation(
            raw_adc=request.raw_adc,
            temperature_c=request.temperature_c,
            terrain_index=request.terrain_index,
            forecast_hours=request.forecast_hours,
            rain_probability_8h=request.rain_probability_8h,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # A missing local sensor position must not change or block the original prediction.
    gis_payload = get_gis_payload_for_sensor(request.sensor_id)
    experimental_score = calculate_experimental_gis_aware_score(
        raw_adc=request.raw_adc,
        estimated_etc=result["estimated_etc"],
        rain_probability_8h=request.rain_probability_8h,
        gis_payload=gis_payload,
    )
    return PredictionResponse(
        sensor_id=request.sensor_id,
        raw_adc=request.raw_adc,
        temperature_c=request.temperature_c,
        terrain_index=request.terrain_index,
        forecast_hours=request.forecast_hours,
        gis_used_for_prediction=False,
        model_validation_status="synthetic_training_data_not_validated_on_real_measurements",
        **result,
        **gis_payload,
        **experimental_score,
    )
