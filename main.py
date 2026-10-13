"""
Terrace Irrigation API
======================

FastAPI server for the terrace irrigation prototype.

Responsibilities:
- Receive sensor data from the Arduino UNO R4 WiFi.
- Authenticate requests using an API key.
- Validate incoming data.
- Run the irrigation ML pipeline.
- Return predicted moisture and irrigation-need score.

The Arduino remains responsible for relay and pump safety.
"""

import os
import secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from irrigation_pipeline import (
    MODEL_VERSION,
    predict_irrigation,
)


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Terrace Irrigation API",
    description=(
        "ML-assisted irrigation predictions using soil "
        "moisture, temperature, terrain and rainfall context."
    ),
    version="1.0.0",
)


# ============================================================
# 2. API-KEY AUTHENTICATION
# ============================================================

# The secret is configured in Render's environment variables.
# Never put the actual API key directly into this source file.

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def require_api_key(
    api_key: str | None = Depends(api_key_header),
) -> None:
    """
    Protect prediction requests using X-API-Key.

    If the server's key has not been configured, reject the
    request rather than exposing the prediction endpoint.
    """

    expected_key = os.getenv("IRRIGATION_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if (
        not api_key
        or not secrets.compare_digest(api_key, expected_key)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


# ============================================================
# 3. REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):
    """
    Incoming data expected from the Arduino.

    terrain_index must already be normalized to [0, 1].
    Rain probability is expressed as a fraction, not a percentage.
    """

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
        description="Normalized terrain index.",
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
# 4. RESPONSE SCHEMA
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


# ============================================================
# 5. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """
    Public health endpoint for Render.

    This endpoint does not expose the API key or sensor data.
    """

    return {
        "status": "ok",
        "service": "Terrace Irrigation API",
        "model_version": MODEL_VERSION,
    }


# ============================================================
# 6. ROOT ENDPOINT
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
    }


# ============================================================
# 7. IRRIGATION PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    api_key: str | None = Depends(api_key_header),
) -> PredictionResponse:
    """
    Authenticate the caller, validate the request and execute
    the irrigation prediction pipeline.
    """

    # Authenticate before running the model.
    require_api_key(api_key)

    try:
        # The ML pipeline performs:
        # 1. ADC calibration
        # 2. Crop water-demand estimation
        # 3. Random Forest moisture prediction
        # 4. Rainfall adjustment
        # 5. Irrigation-need score calculation

        result = predict_irrigation(
            raw_adc=request.raw_adc,
            temperature_c=request.temperature_c,
            terrain_index=request.terrain_index,
            forecast_hours=request.forecast_hours,
            rain_probability_8h=request.rain_probability_8h,
        )

    except ValueError as exc:
        # Invalid pipeline inputs should produce a client error.
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Add the sensor identifier to the pipeline result.
    response = PredictionResponse(
        sensor_id=request.sensor_id,
        **result,
    )

    return response