"""DEM → Mapbox terrain-RGB 타일 (deck.gl TerrainLayer용). 90m DEM을 쌍선형으로 재투영."""
import io
import math
from functools import lru_cache

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

from src.static_data.dem import DEM_CLIP

R = 6378137.0
ORIGIN = math.pi * R


def tile_bounds_3857(z, x, y):
    size = 2 * ORIGIN / 2 ** z
    return (-ORIGIN + x * size, ORIGIN - (y + 1) * size, -ORIGIN + (x + 1) * size, ORIGIN - y * size)


@lru_cache(maxsize=1)
def _src():
    with rasterio.open(DEM_CLIP) as s:
        arr = s.read(1).astype("float32")
        arr[arr == s.nodata] = np.nan
        return arr, s.transform, s.crs


def _encode_png(elev: np.ndarray) -> bytes:
    from PIL import Image
    v = np.round((np.nan_to_num(elev, nan=0.0) + 10000) * 10).astype(np.uint32)
    rgb = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "PNG")
    return buf.getvalue()


@lru_cache(maxsize=512)
def terrain_tile(z: int, x: int, y: int) -> bytes:
    arr, transform, crs = _src()
    dst = np.full((256, 256), np.nan, dtype="float32")
    reproject(arr, dst, src_transform=transform, src_crs=crs, src_nodata=np.nan,
              dst_transform=from_bounds(*tile_bounds_3857(z, x, y), 256, 256), dst_crs="EPSG:3857",
              dst_nodata=np.nan, resampling=Resampling.bilinear)
    return _encode_png(dst)
