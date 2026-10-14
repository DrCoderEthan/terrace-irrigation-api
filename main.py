"""
Terrace Irrigation API
======================

FastAPI server for the terrace irrigation prototype.

Responsibilities:
- Receive sensor data from the Arduino UNO R4 WiFi.
- Authenticate requests using an API key.
- Validate incoming data.
- Run the existing irrigation pipeline.
- Return the original prediction plus GIS terrain metadata.

GIS features are currently for monitoring/verification only. They do
not alter the Random Forest prediction or local Arduino pump control.
The Arduino remains responsible for relay and pump safety.
"""

import logging
import os
import secrets

from fastapi import Depends, FastAPI, HTTPException
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
)


logger = logging.getLogger(__name__)


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Terrace Irrigation API",
    description=(
        "ML-assisted irrigation predictions using soil moisture, "
        "temperature, terrain and rainfall context. GIS terrain "
        "features are currently returned for verification and do "
        "not yet affect the Random Forest prediction."
    ),
    version="1.1.0",
)


# ============================================================
# 2. API-KEY AUTHENTICATION
# ============================================================

# Configure the secret in Render's environment variables.
# Never put the actual API key directly into this source file.

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def require_api_key(
    api_key: str | None = Depends(api_key_header),
) -> None:
    """Protect endpoints using the X-API-Key header."""

    expected_key = os.getenv("IRRIGATION_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


# ============================================================
# 3. GIS LOOKUP
# ============================================================

def get_gis_payload_for_sensor(sensor_id: str) -> dict:
    """Return GIS features for a configured sensor location.

    For now only terrace_sensor_01 is configured by convention.
    Its actual coordinates must be set using TERRACE_SENSOR_LAT and
    TERRACE_SENSOR_LON in the hosting environment. Without coordinates,
    GIS is marked unavailable and does not break the existing prediction.
    """

    def unavailable(status: str, message: str) -> dict:
        return {
            "gis_status": status,
            "gis_message": message,
            "gis_location": None,
            "terrain_features": None,
        }

    if sensor_id != "terrace_sensor_01":
        return unavailable(
            "location_not_configured",
            "No GIS location is configured for this sensor ID.",
        )

    latitude_text = os.getenv("TERRACE_SENSOR_LAT")
    longitude_text = os.getenv("TERRACE_SENSOR_LON")

    if not latitude_text and not longitude_text:
        return unavailable(
            "location_not_configured",
            "Set TERRACE_SENSOR_LAT and TERRACE_SENSOR_LON to the actual sensor coordinates.",
        )

    if not latitude_text or not longitude_text:
        return unavailable(
            "configuration_error",
            "Both TERRACE_SENSOR_LAT and TERRACE_SENSOR_LON must be configured.",
        )

    try:
        latitude = float(latitude_text)
        longitude = float(longitude_text)

        sample = get_terrain_sample(
            latitude=latitude,
            longitude=longitude,
        )

        return {
            "gis_status": "ok",
            "gis_message": None,
            "gis_location": sample["location"],
            "terrain_features": sample["features"],
        }

    except LocationOutsideRasterError:
        return unavailable(
            "outside_raster",
            "The configured sensor location is outside the terrain raster.",
        )

    except NoTerrainDataError:
        return unavailable(
            "no_data",
            "The terrain raster contains missing data at the configured sensor location.",
        )

    except GISFeatureError:
        logger.exception("GIS feature extraction failed for sensor %s", sensor_id)
        return unavailable(
            "unavailable",
            "GIS data could not be read. The existing prediction was not changed.",
        )

    except (ValueError, TypeError):
        return unavailable(
            "configuration_error",
            "The configured sensor coordinates are invalid.",
        )

    except Exception:
        logger.exception("Unexpected GIS error for sensor %s", sensor_id)
        return unavailable(
            "error",
            "An unexpected GIS error occurred. The existing prediction was not changed.",
        )


# ============================================================
# 4. REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):
    """Incoming data expected from the Arduino."""

    sensor_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Unique identifier for the soil sensor.",
    )

    raw_adc: int = Field(
        ge=0,
        le=4095,
        description="Raw ADC reading from the moisture sensor.",
    )

    temperature_c: float = Field(
        ge=-10,
        le=60,
        allow_inf_nan=False,
        description="Temperature in degrees Celsius.",
    )

    terrain_index: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
        description=(
            "Legacy normalized terrain input. It is still passed to the "
            "existing model and is not replaced with raw GIS TWI."
        ),
    )

    forecast_hours: float = Field(
        ge=6,
        le=8,
        allow_inf_nan=False,
        description="Prediction horizon in hours.",
    )

    rain_probability_8h: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="Probability of rain over the next eight hours.",
    )


# ============================================================
# 5. RESPONSE SCHEMA
# ============================================================

class PredictionResponse(BaseModel):
    """Response returned to the requesting device."""

    sensor_id: str
    current_moisture: float
    predicted_moisture: float
    estimated_etc: float
    rain_probability_8h: float
    irrigation_need_score: float
    recommendation: str
    model_version: str

    # GIS details are additional monitoring fields; they do not
    # currently affect predicted_moisture or irrigation_need_score.
    gis_status: str = "location_not_configured"
    gis_message: str | None = None
    gis_location: dict[str, float] | None = None
    terrain_features: dict[str, float | None] | None = None


# ============================================================
# 6. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """Public health endpoint for Render."""

    return {
        "status": "ok",
        "service": "Terrace Irrigation API",
        "model_version": MODEL_VERSION,
    }


# ============================================================
# 7. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """Basic service information."""

    return {
        "service": "Terrace Irrigation API",
        "status": "running",
        "health_endpoint": "/health",
        "documentation": "/docs",
        "prediction_endpoint": "/predict",
        "gis_info_endpoint": "/gis/info",
        "gis_test_endpoint": "/gis/test-zone/{zone_code}",
    }


# ============================================================
# 8. GIS TEST ENDPOINTS
# ============================================================

@app.get("/gis/info")
def gis_info(
    api_key: str | None = Depends(api_key_header),
):
    """Return terrain raster metadata. Requires the API key."""

    require_api_key(api_key)

    try:
        return get_raster_info()
    except GISFeatureError as exc:
        logger.exception("Could not read GIS raster.")
        raise HTTPException(
            status_code=503,
            detail="The GIS raster is unavailable.",
        ) from exc


@app.get("/gis/test-zone/{zone_code}")
def gis_test_zone(
    zone_code: str,
    api_key: str | None = Depends(api_key_header),
):
    """Test terrain extraction at a coordinate in zones.csv.

    This is a dataset test location, not an assertion that the selected
    zone contains the actual irrigation sensor.
    """

    require_api_key(api_key)

    try:
        return get_terrain_sample_for_zone(zone_code.upper())
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except GISFeatureError as exc:
        logger.exception("GIS extraction failed for test zone %s", zone_code)
        raise HTTPException(
            status_code=503,
            detail="Terrain extraction failed.",
        ) from exc


# ============================================================
# 9. IRRIGATION PREDICTION ENDPOINT
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    api_key: str | None = Depends(api_key_header),
) -> PredictionResponse:
    """Authenticate, validate, run the existing pipeline, and append GIS data."""

    require_api_key(api_key)

    try:
        # Keep the existing ML call unchanged for this integration stage.
        result = predict_irrigation(
            raw_adc=request.raw_adc,
            temperature_c=request.temperature_c,
            terrain_index=request.terrain_index,
            forecast_hours=request.forecast_hours,
            rain_probability_8h=request.rain_probability_8h,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # GIS is supplementary and cannot prevent the existing prediction
    # from being returned if coordinates or raster features are unavailable.
    gis_payload = get_gis_payload_for_sensor(request.sensor_id)

    return PredictionResponse(
        sensor_id=request.sensor_id,
        **result,
        **gis_payload,
    )
