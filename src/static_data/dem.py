"""DEM 기반 지형 피처: 지반고(GL), 주변 평균 대비 상대고도, 경사.

지형 자료는 두 가지 중 하나를 쓴다.
- 기본: 국토지리정보원 공개 DEM 37612 도엽 (EPSG:5179, 90m) — settings.dem_path
- DEM_DIR 지정 시: 그 폴더의 5m급 자료
  - 래스터(.tif/.tiff/.img) 여러 도엽 → 이어 붙임
  - 수치지형도 등고선·표고점(.shp/.gpkg/.dxf) → 선형 보간(TIN)으로 격자 생성
값은 쌍선형 보간으로 뽑고, 결과에 해상도를 함께 기록한다.
"""
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from scipy.signal import fftconvolve

from src.config import PROCESSED, ROOT, settings

log = logging.getLogger(__name__)
DEM_CLIP = PROCESSED / "dem_sillim.tif"
RASTER_EXT = {".tif", ".tiff", ".img"}
VECTOR_EXT = {".shp", ".gpkg", ".dxf"}
# 수치지형도 등고선·표고점의 높이 필드 후보 (파일마다 다를 수 있어 로그로 확인)
HEIGHT_FIELDS = ["CONT", "ELEV", "ELEVATION", "HEIGHT", "Z", "등고수치", "표고수치", "수치"]


class Dem:
    def __init__(self, arr: np.ndarray, transform, crs, nodata):
        self.arr = arr.astype("float64")
        if nodata is not None:
            self.arr[arr == nodata] = np.nan
        self.transform = transform
        self.crs = crs
        self.res = transform.a

    # ---------- 만들기 ----------
    @classmethod
    def build_clip(cls, bounds, bounds_crs, margin_m=1500):
        """신림동 범위(bounds, bounds_crs 좌표)에 여유를 둔 DEM을 만들어 DEM_CLIP에 저장."""
        dem_dir = Path(settings.dem_dir) if settings.dem_dir else None
        if dem_dir and not dem_dir.is_absolute():
            dem_dir = ROOT / dem_dir
        if dem_dir:
            files = sorted(p for p in dem_dir.rglob("*") if p.suffix.lower() in RASTER_EXT | VECTOR_EXT)
            rasters = [p for p in files if p.suffix.lower() in RASTER_EXT]
            vectors = [p for p in files if p.suffix.lower() in VECTOR_EXT]
            if rasters:
                dem = cls._from_rasters(rasters, bounds, bounds_crs, margin_m)
            elif vectors:
                dem = cls._from_contours(vectors, bounds, bounds_crs, margin_m)
            else:
                raise FileNotFoundError(f"DEM_DIR({dem_dir})에 래스터·등고선 파일이 없음")
        else:
            dem = cls._from_rasters([Path(settings.dem_path)], bounds, bounds_crs, margin_m)
        dem.save(DEM_CLIP)
        log.info("지형 격자 저장: %s (%d x %d, %.1f m, CRS=%s, 결측 %d칸)", DEM_CLIP.name, dem.arr.shape[1],
                 dem.arr.shape[0], dem.res, dem.crs, int(np.isnan(dem.arr).sum()))
        return dem

    @classmethod
    def _from_rasters(cls, paths, bounds, bounds_crs, margin_m):
        from pyproj import Transformer
        with rasterio.open(paths[0]) as first:
            crs = first.crs
            log.info("DEM 래스터 %d개, 첫 파일 %s: CRS=%s 해상도=%s nodata=%s", len(paths), paths[0].name,
                     crs, first.res, first.nodata)
        x0, y0, x1, y1 = Transformer.from_crs(bounds_crs, crs, always_xy=True).transform_bounds(*bounds)
        box = (x0 - margin_m, y0 - margin_m, x1 + margin_m, y1 + margin_m)
        if len(paths) == 1:
            with rasterio.open(paths[0]) as src:
                win = from_bounds(*box, src.transform)
                # 창이 DEM 경계를 넘으면 읽기는 잘리는데 transform은 음수 오프셋을 유지해 어긋나므로 교집합으로 제한
                win = win.round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
                return cls(src.read(1, window=win), src.window_transform(win), crs, src.nodata)
        srcs = []
        try:
            for p in paths:
                s = rasterio.open(p)
                if s.crs != crs:  # 도엽마다 좌표계가 다르면 첫 파일 좌표계로 맞춤
                    log.warning("%s 좌표계 %s → %s 로 변환", p.name, s.crs, crs)
                    s = WarpedVRT(s, crs=crs)
                srcs.append(s)
            nodata = srcs[0].nodata if srcs[0].nodata is not None else -9999.0
            arr, transform = merge(srcs, bounds=box, nodata=nodata)
        finally:
            for s in srcs:
                s.close()
        dem = cls(arr[0], transform, crs, nodata)
        if np.isnan(dem.arr).any():
            log.warning("이어 붙인 DEM에 빈 칸 %d개 — 받은 도엽이 신림동 범위를 다 덮는지 확인 필요",
                        int(np.isnan(dem.arr).sum()))
        return dem

    @classmethod
    def _from_contours(cls, paths, bounds, bounds_crs, margin_m, res=5.0):
        """등고선 정점·표고점 → 5m 격자 선형 보간 (scipy griddata, Delaunay TIN)."""
        import geopandas as gpd
        from scipy.interpolate import griddata
        pts, vals, crs = [], [], None
        for p in paths:
            g = gpd.read_file(p)
            crs = crs or g.crs
            if g.crs is None:
                raise ValueError(f"{p.name} 좌표계 정보 없음 — .prj 파일 확인 필요")
            g = g.to_crs(crs)
            field = next((f for f in HEIGHT_FIELDS if f in g.columns), None)
            log.info("등고선 파일 %s: %d개 객체, 필드=%s, 높이 필드=%s", p.name, len(g), list(g.columns), field)
            for geom, h in zip(g.geometry, g[field] if field else [None] * len(g)):
                if geom is None or geom.is_empty:
                    continue
                coords = [c for part in getattr(geom, "geoms", [geom]) for c in
                          (part.coords if hasattr(part, "coords") else part.exterior.coords)]
                for c in coords:
                    z = h if h is not None else (c[2] if len(c) > 2 else None)  # 필드 없으면 3D 좌표의 Z (DXF)
                    if z is None:
                        continue
                    pts.append(c[:2])
                    vals.append(float(z))
        if not pts:
            raise ValueError("등고선에서 높이 값을 찾지 못함 — 높이 필드명을 HEIGHT_FIELDS에 추가 필요")
        from pyproj import Transformer
        x0, y0, x1, y1 = Transformer.from_crs(bounds_crs, crs, always_xy=True).transform_bounds(*bounds)
        x0, y0, x1, y1 = x0 - margin_m, y0 - margin_m, x1 + margin_m, y1 + margin_m
        xs = np.arange(x0 + res / 2, x1, res)
        ys = np.arange(y1 - res / 2, y0, -res)
        gx, gy = np.meshgrid(xs, ys)
        pts, vals = np.asarray(pts), np.asarray(vals)
        log.info("등고선 정점 %d개 → %d x %d 격자 보간", len(pts), len(xs), len(ys))
        grid = griddata(pts, vals, (gx, gy), method="linear")
        return cls(grid, from_origin(x0, y1, res, res), crs, None)

    def save(self, path: Path):
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out = np.where(np.isnan(self.arr), -9999.0, self.arr).astype("float32")
        with rasterio.open(path, "w", driver="GTiff", width=out.shape[1], height=out.shape[0], count=1,
                           dtype="float32", crs=self.crs, transform=self.transform, nodata=-9999.0,
                           compress="lzw", predictor=3, tiled=True) as dst:
            dst.write(out, 1)

    @classmethod
    def load_clip(cls):
        with rasterio.open(DEM_CLIP) as src:
            return cls(src.read(1), src.transform, src.crs, src.nodata)

    # ---------- 계산 ----------
    def _rowcol(self, x, y):
        inv = ~self.transform
        c, r = inv * (np.asarray(x), np.asarray(y))
        return r - 0.5, c - 0.5  # 셀 중심 기준

    def sample(self, grid: np.ndarray, x, y) -> np.ndarray:
        """grid(=DEM 또는 파생 격자)를 좌표 x, y에서 쌍선형 보간. 범위 밖은 NaN."""
        r, c = self._rowcol(x, y)
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        h, w = grid.shape
        ok = (r0 >= 0) & (c0 >= 0) & (r0 + 1 < h) & (c0 + 1 < w)
        out = np.full(r.shape, np.nan)
        r0, c0, fr, fc = r0[ok], c0[ok], fr[ok], fc[ok]
        out[ok] = (grid[r0, c0] * (1 - fr) * (1 - fc) + grid[r0, c0 + 1] * (1 - fr) * fc
                   + grid[r0 + 1, c0] * fr * (1 - fc) + grid[r0 + 1, c0 + 1] * fr * fc)
        return out

    def focal_mean(self, radius_m: float) -> np.ndarray:
        """반경 radius_m 원형 창 평균 (NaN 무시). 5m 격자에서는 창이 1만 칸이 넘어 FFT 합성곱으로 계산한다."""
        k = int(np.ceil(radius_m / self.res))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        kernel = ((xx * xx + yy * yy) * self.res ** 2 <= radius_m ** 2).astype(float)
        valid = ~np.isnan(self.arr)
        acc = fftconvolve(np.where(valid, self.arr, 0.0), kernel, mode="same")
        cnt = fftconvolve(valid.astype(float), kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = acc / np.round(cnt, 6)
        out[np.round(cnt) < 1] = np.nan
        return out

    def slope_pct(self) -> np.ndarray:
        """Horn(1981) 방식 경사(%)."""
        z = np.pad(self.arr, 1, mode="edge")
        a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
        d, f = z[1:-1, :-2], z[1:-1, 2:]
        g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
        dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * self.res)
        dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * self.res)
        return np.hypot(dzdx, dzdy) * 100
