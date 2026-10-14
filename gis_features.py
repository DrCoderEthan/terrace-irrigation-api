"""GIS feature extraction for the cardboard terrace terrain model.

The model is indexed in local board coordinates, not GPS coordinates.
Origin: top-left corner. X increases left-to-right; Y increases top-to-bottom.
Default board extent: 40 cm (X) x 30 cm (Y), based on dem_points.csv.

The raster's projected CRS is retained as raster metadata. Sensor lookup uses
local centimetres and converts them to raster row/column indices.
"""

from __future__ import annotations

import csv
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.windows import Window

BASE_DIR = Path(__file__).resolve().parent
GIS_DIR = Path(os.getenv("TERRAIN_GIS_DIR", str(BASE_DIR / "gis")))
RASTER_PATH = Path(
    os.getenv("TERRAIN_RASTER_PATH", str(GIS_DIR / "terrain_stack.tif"))
)
ZONES_CSV_PATH = Path(
    os.getenv("TERRAIN_ZONES_PATH", str(GIS_DIR / "zones.csv"))
)

# The CSV's board_x_cm/board_y_cm values span cell centres from 0.05 to 39.95
# and 0.05 to 29.95, giving a 40 x 30 cm local model.
BOARD_WIDTH_CM = float(os.getenv("TERRAIN_BOARD_WIDTH_CM", "40.0"))
BOARD_HEIGHT_CM = float(os.getenv("TERRAIN_BOARD_HEIGHT_CM", "30.0"))

BAND_NAMES = (
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "curvature",
    "tpi_m",
    "twi",
)


class GISFeatureError(RuntimeError):
    """Base error for GIS raster loading or feature extraction."""


class LocationOutsideRasterError(GISFeatureError):
    """A local model coordinate or test point falls outside the raster."""


class NoTerrainDataError(GISFeatureError):
    """The selected raster cell has missing or invalid feature values."""


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _validate_dataset(dataset: rasterio.io.DatasetReader) -> None:
    if dataset.crs is None:
        raise GISFeatureError("Terrain raster has no CRS metadata.")
    if dataset.count != len(BAND_NAMES):
        raise GISFeatureError(
            f"Expected {len(BAND_NAMES)} terrain bands, found {dataset.count}."
        )
    descriptions = tuple(desc or "" for desc in dataset.descriptions)
    if any(descriptions) and descriptions != BAND_NAMES:
        raise GISFeatureError(
            "Terrain band order differs from the expected order: "
            f"{descriptions!r}."
        )
    if BOARD_WIDTH_CM <= 0 or BOARD_HEIGHT_CM <= 0:
        raise GISFeatureError("Board dimensions must be positive.")


def _read_cell(
    dataset: rasterio.io.DatasetReader,
    row: int,
    col: int,
    *,
    requested_x_cm: float | None = None,
    requested_y_cm: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not (0 <= row < dataset.height and 0 <= col < dataset.width):
        raise LocationOutsideRasterError(
            "The selected point falls outside the terrain raster."
        )

    pixel = dataset.read(
        window=Window(col_off=col, row_off=row, width=1, height=1),
        masked=True,
    )[:, 0, 0]

    features: dict[str, float | None] = {}
    for name, raw in zip(BAND_NAMES, pixel):
        if np.ma.is_masked(raw):
            raise NoTerrainDataError(f"Missing raster data in band '{name}'.")
        value = float(raw)
        if not math.isfinite(value):
            raise NoTerrainDataError(f"Invalid raster value in band '{name}'.")
        if dataset.nodata is not None and value == float(dataset.nodata):
            raise NoTerrainDataError(f"No-data value in band '{name}'.")
        # -1 marks flat/undefined aspect in this dataset; do not treat it as a bearing.
        if name == "aspect_deg" and value < 0:
            features[name] = None
        else:
            features[name] = round(value, 4)

    center_x_cm = (col + 0.5) * (BOARD_WIDTH_CM / dataset.width)
    center_y_cm = (row + 0.5) * (BOARD_HEIGHT_CM / dataset.height)

    result: dict[str, Any] = {
        "model_location": {
            "x_cm": round(center_x_cm, 4),
            "y_cm": round(center_y_cm, 4),
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
        },
        "requested_model_location": (
            {
                "x_cm": round(requested_x_cm, 4),
                "y_cm": round(requested_y_cm, 4),
            }
            if requested_x_cm is not None and requested_y_cm is not None
            else None
        ),
        "raster": {
            "source": Path(dataset.name).name,
            "crs_metadata": str(dataset.crs),
            "pixel_row": int(row),
            "pixel_col": int(col),
            "pixel_center_x_cm": round(center_x_cm, 4),
            "pixel_center_y_cm": round(center_y_cm, 4),
            "board_width_cm": BOARD_WIDTH_CM,
            "board_height_cm": BOARD_HEIGHT_CM,
            "cell_width_cm": round(BOARD_WIDTH_CM / dataset.width, 6),
            "cell_height_cm": round(BOARD_HEIGHT_CM / dataset.height, 6),
        },
        "features": features,
    }
    if extra:
        result.update(extra)
    return result


def get_raster_info(raster_path: str | Path | None = None) -> dict[str, Any]:
    """Read terrain raster metadata, including local board extent."""
    path = Path(raster_path) if raster_path else RASTER_PATH
    if not path.is_file():
        raise GISFeatureError(f"Terrain raster not found: {path}")

    try:
        with rasterio.open(path) as dataset:
            _validate_dataset(dataset)
            return {
                "source": path.name,
                "crs_metadata": str(dataset.crs),
                "width_px": dataset.width,
                "height_px": dataset.height,
                "bands": list(BAND_NAMES),
                "nodata": dataset.nodata,
                "coordinate_mode": "local_cardboard_model_cm",
                "origin": "top_left",
                "x_direction": "right",
                "y_direction": "down",
                "board_width_cm": BOARD_WIDTH_CM,
                "board_height_cm": BOARD_HEIGHT_CM,
                "cell_width_cm": round(BOARD_WIDTH_CM / dataset.width, 6),
                "cell_height_cm": round(BOARD_HEIGHT_CM / dataset.height, 6),
                "note": (
                    "CRS is raster metadata; sensor lookup uses local board coordinates, "
                    "not latitude/longitude. Verify board dimensions and axis orientation."
                ),
            }
    except RasterioError as exc:
        raise GISFeatureError(f"Could not open terrain raster: {exc}") from exc


def get_terrain_sample(
    x_cm: float,
    y_cm: float,
    raster_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract terrain features at local board coordinates in centimetres.

    (0, 0) is the top-left corner. X increases right; Y increases down.
    Values at the exact right/bottom boundary map to the last cell.
    """
    x_cm = _finite_float(x_cm, "x_cm")
    y_cm = _finite_float(y_cm, "y_cm")
    if not (0 <= x_cm <= BOARD_WIDTH_CM):
        raise LocationOutsideRasterError(
            f"x_cm must be between 0 and {BOARD_WIDTH_CM} cm."
        )
    if not (0 <= y_cm <= BOARD_HEIGHT_CM):
        raise LocationOutsideRasterError(
            f"y_cm must be between 0 and {BOARD_HEIGHT_CM} cm."
        )

    path = Path(raster_path) if raster_path else RASTER_PATH
    if not path.is_file():
        raise GISFeatureError(f"Terrain raster not found: {path}")

    try:
        with rasterio.open(path) as dataset:
            _validate_dataset(dataset)
            col = min(int(math.floor(x_cm / BOARD_WIDTH_CM * dataset.width)), dataset.width - 1)
            row = min(int(math.floor(y_cm / BOARD_HEIGHT_CM * dataset.height)), dataset.height - 1)
            return _read_cell(
                dataset,
                row,
                col,
                requested_x_cm=x_cm,
                requested_y_cm=y_cm,
            )
    except RasterioError as exc:
        raise GISFeatureError(f"Could not read terrain raster: {exc}") from exc


def get_terrain_features(
    x_cm: float,
    y_cm: float,
    raster_path: str | Path | None = None,
) -> dict[str, float | None]:
    """Return only the six terrain features for a local model coordinate."""
    return get_terrain_sample(x_cm, y_cm, raster_path)["features"]



def get_terrain_feature_ranges(
    raster_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Return valid-cell min/max ranges for slope and TWI.

    Ranges are specific to this supplied terrain raster. They are used only to
    normalize an explicitly experimental, local-model scoring heuristic, not
    to claim universal physical meaning for either feature. Results are cached
    per raster path so the full raster is not read on each API request.
    """
    path = Path(raster_path) if raster_path else RASTER_PATH
    return _cached_terrain_feature_ranges(str(path.resolve()))


@lru_cache(maxsize=4)
def _cached_terrain_feature_ranges(path_string: str) -> dict[str, dict[str, float]]:
    path = Path(path_string)
    if not path.is_file():
        raise GISFeatureError(f"Terrain raster not found: {path}")

    try:
        with rasterio.open(path) as dataset:
            _validate_dataset(dataset)
            result: dict[str, dict[str, float]] = {}
            for feature_name, band_index in (("slope_deg", 2), ("twi", 6)):
                masked = dataset.read(band_index, masked=True).astype(np.float64)
                values = np.asarray(masked.compressed(), dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    raise NoTerrainDataError(
                        f"No valid raster cells available for '{feature_name}'."
                    )
                result[feature_name] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            return result
    except RasterioError as exc:
        raise GISFeatureError(f"Could not read terrain feature ranges: {exc}") from exc

def get_terrain_sample_for_zone(
    zone_code: str,
    raster_path: str | Path | None = None,
    zones_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Test a named dataset zone and map it back to local board coordinates.

    zones.csv easting/northing are used only to identify the corresponding
    cell in this raster's internal georeferencing. The response reports local
    board centimetres and does not claim these values are real-world GPS.
    """
    code = str(zone_code).strip().upper()
    if not code:
        raise ValueError("Zone code cannot be empty.")

    csv_path = Path(zones_csv_path) if zones_csv_path else ZONES_CSV_PATH
    raster = Path(raster_path) if raster_path else RASTER_PATH
    if not csv_path.is_file():
        raise GISFeatureError(f"Zones CSV not found: {csv_path}")
    if not raster.is_file():
        raise GISFeatureError(f"Terrain raster not found: {raster}")

    zone_row: dict[str, str] | None = None
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("zone", "").strip().upper() == code:
                    zone_row = row
                    break
    except OSError as exc:
        raise GISFeatureError(f"Could not read zones CSV: {exc}") from exc

    if zone_row is None:
        raise ValueError(f"Zone '{code}' was not found in {csv_path.name}.")

    try:
        easting = _finite_float(zone_row.get("easting"), "zone easting")
        northing = _finite_float(zone_row.get("northing"), "zone northing")
        with rasterio.open(raster) as dataset:
            _validate_dataset(dataset)
            row, col = dataset.index(easting, northing)
            sample = _read_cell(dataset, row, col)
    except RasterioError as exc:
        raise GISFeatureError(f"Could not read terrain raster: {exc}") from exc

    sample["dataset_test_zone"] = {
        "code": code,
        "name": zone_row.get("name", "").strip(),
        "note": "Test point from zones.csv; not the configured physical sensor location.",
    }
    return sample


if __name__ == "__main__":
    print("--- Local terrain raster information ---")
    print(get_raster_info())
    print("\n--- Sample at X=16.55 cm, Y=12.05 cm (test only) ---")
    print(get_terrain_sample(16.55, 12.05))
    print("\n--- Dataset zone A (test only) ---")
    print(get_terrain_sample_for_zone("A"))
