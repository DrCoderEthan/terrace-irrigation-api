"""
Terrace Irrigation Project — GIS Feature Extraction

Loads terrain_stack.tif and extracts terrain features for a
specified geographic location.

Expected project structure:
    main.py
    irrigation_pipeline.py
    gis_features.py
    requirements.txt
    gis/
        terrain_stack.tif
        zones.csv

IMPORTANT:
GIS features are extracted for analysis. They are not yet
validated irrigation recommendations or ML predictions.
"""

import csv
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window
from rasterio.errors import RasterioError


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RASTER_PATH = Path(
    os.getenv(
        "TERRAIN_RASTER_PATH",
        str(BASE_DIR / "gis" / "terrain_stack.tif")
    )
)

ZONES_CSV_PATH = Path(
    os.getenv(
        "TERRAIN_ZONES_PATH",
        str(BASE_DIR / "gis" / "zones.csv")
    )
)

# Band order verified against the supplied terrain_stack.tif.
BAND_NAMES = (
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "curvature",
    "tpi_m",
    "twi"
)


# ============================================================
# 2. GIS ERRORS
# ============================================================

class GISFeatureError(RuntimeError):
    """Base error for GIS loading or feature extraction."""


class LocationOutsideRasterError(GISFeatureError):
    """The requested location is outside the raster."""


class NoTerrainDataError(GISFeatureError):
    """The requested raster pixel has missing or invalid data."""


# ============================================================
# 3. INPUT VALIDATION
# ============================================================

def _finite_float(value, name):
    """Convert a value to a finite float."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a valid number."
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _validate_raster(dataset):
    """Verify that the loaded raster matches our expected stack."""

    if dataset.crs is None:
        raise GISFeatureError(
            "The terrain raster has no coordinate reference system."
        )

    if dataset.count != len(BAND_NAMES):
        raise GISFeatureError(
            f"Expected 6 raster bands; found {dataset.count}."
        )

    # When the file supplies band descriptions, check their order.
    descriptions = tuple(
        description or ""
        for description in dataset.descriptions
    )

    if any(descriptions) and descriptions != BAND_NAMES:
        raise GISFeatureError(
            "Terrain band order does not match the expected "
            f"order. Found: {descriptions}"
        )


# ============================================================
# 4. RASTER INFORMATION
# ============================================================

def get_raster_info(raster_path=None):
    """
    Return raster metadata without loading the entire raster
    into memory.
    """

    path = Path(raster_path) if raster_path else RASTER_PATH

    if not path.is_file():
        raise GISFeatureError(
            f"Terrain raster not found: {path}"
        )

    try:
        with rasterio.open(path) as dataset:
            _validate_raster(dataset)

            return {
                "source": path.name,
                "crs": str(dataset.crs),
                "width_px": dataset.width,
                "height_px": dataset.height,
                "pixel_size_x": abs(
                    float(dataset.transform.a)
                ),
                "pixel_size_y": abs(
                    float(dataset.transform.e)
                ),
                "bounds": {
                    "left": float(dataset.bounds.left),
                    "bottom": float(dataset.bounds.bottom),
                    "right": float(dataset.bounds.right),
                    "top": float(dataset.bounds.top)
                },
                "bands": list(BAND_NAMES),
                "nodata": dataset.nodata
            }

    except RasterioError as exc:
        raise GISFeatureError(
            f"Could not open terrain raster: {exc}"
        ) from exc


# ============================================================
# 5. EXTRACT TERRAIN AT A GEOGRAPHIC LOCATION
# ============================================================

def get_terrain_sample(
    latitude,
    longitude,
    raster_path=None
):
    """
    Extract all six terrain features at a geographic location.

    Parameters
    ----------
    latitude:
        Latitude in decimal degrees.

    longitude:
        Longitude in decimal degrees.

    raster_path:
        Optional path to a different terrain raster.

    Returns
    -------
    Dictionary containing:
        - Geographic location
        - Raster metadata
        - Six terrain features
    """

    latitude = _finite_float(latitude, "latitude")
    longitude = _finite_float(longitude, "longitude")

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    path = Path(raster_path) if raster_path else RASTER_PATH

    if not path.is_file():
        raise GISFeatureError(
            f"Terrain raster not found: {path}"
        )

    try:
        with rasterio.open(path) as dataset:

            _validate_raster(dataset)

            # Transform WGS84 longitude/latitude into the
            # coordinate reference system of the raster.
            try:
                xs, ys = transform(
                    "EPSG:4326",
                    dataset.crs,
                    [longitude],
                    [latitude]
                )
            except Exception as exc:
                raise ValueError(
                    "Could not transform the supplied geographic "
                    "coordinates into the raster coordinate system."
                ) from exc

            x = float(xs[0])
            y = float(ys[0])

            # Convert map coordinates into a raster row and column.
            row, col = dataset.index(x, y)

            # Never silently use an unrelated pixel.
            if not (
                0 <= row < dataset.height
                and 0 <= col < dataset.width
            ):
                raise LocationOutsideRasterError(
                    "The requested location is outside the terrain "
                    "raster. Verify the sensor coordinates and "
                    "raster coverage."
                )

            # Read only the required pixel from all six bands.
            window = Window(
                col_off=col,
                row_off=row,
                width=1,
                height=1
            )

            pixel = dataset.read(
                window=window,
                masked=True
            )[:, 0, 0]

            features = {}

            for band_name, value in zip(BAND_NAMES, pixel):

                if np.ma.is_masked(value):
                    raise NoTerrainDataError(
                        f"Missing data in band '{band_name}'."
                    )

                value = float(value)

                if not math.isfinite(value):
                    raise NoTerrainDataError(
                        f"Invalid value in band '{band_name}'."
                    )

                if (
                    dataset.nodata is not None
                    and value == float(dataset.nodata)
                ):
                    raise NoTerrainDataError(
                        f"No-data value found in band '{band_name}'."
                    )

                features[band_name] = value

            # The supplied raster uses -1 for undefined aspect.
            # Do not interpret -1 as a compass direction.
            if features["aspect_deg"] < 0:
                features["aspect_deg"] = None

            # Convert NumPy values to ordinary Python numbers,
            # making the output JSON-friendly.
            features = {
                name: (
                    round(value, 4)
                    if value is not None
                    else None
                )
                for name, value in features.items()
            }

            center_x, center_y = dataset.xy(row, col)

            return {
                "location": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "raster": {
                    "source": path.name,
                    "crs": str(dataset.crs),
                    "pixel_row": int(row),
                    "pixel_col": int(col),
                    "pixel_center_x": float(center_x),
                    "pixel_center_y": float(center_y)
                },
                "features": features
            }

    except RasterioError as exc:
        raise GISFeatureError(
            f"Could not read terrain raster: {exc}"
        ) from exc


# ============================================================
# 6. GET ONLY THE TERRAIN FEATURES
# ============================================================

def get_terrain_features(
    latitude,
    longitude,
    raster_path=None
):
    """
    Return only terrain feature values.

    This function will be convenient when we connect
    GIS extraction to main.py.
    """

    sample = get_terrain_sample(
        latitude=latitude,
        longitude=longitude,
        raster_path=raster_path
    )

    return sample["features"]


# ============================================================
# 7. OPTIONAL: LOOK UP A TEST LOCATION FROM zones.csv
# ============================================================

def get_zone_coordinates(
    zone_code,
    zones_csv_path=None
):
    """
    Get coordinates for an explicitly selected dataset zone.

    This is for GIS testing. A dataset zone must not be assumed
    to represent the actual sensor location.
    """

    zone_code = str(zone_code).strip().upper()

    if not zone_code:
        raise ValueError("Zone code cannot be empty.")

    path = (
        Path(zones_csv_path)
        if zones_csv_path
        else ZONES_CSV_PATH
    )

    if not path.is_file():
        raise GISFeatureError(
            f"Zone CSV not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as csv_file:

            for row in csv.DictReader(csv_file):

                if row.get("zone", "").strip().upper() == zone_code:

                    return {
                        "code": zone_code,
                        "name": row.get("name", "").strip(),
                        "latitude": _finite_float(
                            row.get("lat"),
                            "zone latitude"
                        ),
                        "longitude": _finite_float(
                            row.get("lon"),
                            "zone longitude"
                        )
                    }

    except OSError as exc:
        raise GISFeatureError(
            f"Could not read zone CSV: {exc}"
        ) from exc

    raise ValueError(
        f"Zone '{zone_code}' was not found in {path.name}."
    )


def get_terrain_sample_for_zone(
    zone_code,
    raster_path=None,
    zones_csv_path=None
):
    """
    Test terrain extraction at a coordinate listed in zones.csv.

    This does not establish that the zone contains your real sensor.
    """

    zone = get_zone_coordinates(
        zone_code,
        zones_csv_path
    )

    result = get_terrain_sample(
        latitude=zone["latitude"],
        longitude=zone["longitude"],
        raster_path=raster_path
    )

    result["zone_test_location"] = {
        "code": zone["code"],
        "name": zone["name"]
    }

    return result


# ============================================================
# 8. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("\n--- GIS RASTER INFORMATION ---")
    print(get_raster_info())

    # Zone A is used only as a dataset test location.
    # It is NOT assumed to be the actual sensor location.

    print("\n--- TEST EXTRACTION: DATASET ZONE A ---")

    result = get_terrain_sample_for_zone("A")

    print(result)