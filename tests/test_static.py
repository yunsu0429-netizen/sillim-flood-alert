import numpy as np
import pytest
from rasterio.transform import from_origin

from src.static_data.dem import Dem


def _loop_focal_mean(arr, res, radius):
    """이전 구현(이중 루프) — FFT 버전과 결과가 같아야 한다."""
    k = int(np.ceil(radius / res))
    vals, valid = np.nan_to_num(arr), (~np.isnan(arr)).astype(float)
    acc, cnt = np.zeros_like(vals), np.zeros_like(vals)
    pv, pm = np.pad(vals, k), np.pad(valid, k)
    h, w = vals.shape
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if (dr * dr + dc * dc) * res ** 2 > radius ** 2:
                continue
            acc += pv[k + dr:k + dr + h, k + dc:k + dc + w]
            cnt += pm[k + dr:k + dr + h, k + dc:k + dc + w]
    with np.errstate(invalid="ignore"):
        return acc / cnt


def test_fft_focal_mean_matches_loop():
    rng = np.random.default_rng(0)
    arr = rng.uniform(10, 60, (40, 50))
    arr[5:8, 10:14] = np.nan                     # 결측 포함
    dem = Dem(arr.copy(), from_origin(0, 400, 10, 10), "EPSG:5186", None)
    fast = dem.focal_mean(35)
    slow = _loop_focal_mean(dem.arr, 10, 35)
    assert np.allclose(fast, slow, equal_nan=True, atol=1e-6)


def test_contour_interpolation_reproduces_points(tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from src.config import settings
    # 서→동으로 10m에서 30m까지 올라가는 경사면의 등고선 3개 + 표고점 1개
    lines = [LineString([(x, 0), (x, 200)]) for x in (0, 100, 200)]
    g = gpd.GeoDataFrame({"CONT": [10.0, 20.0, 30.0]}, geometry=lines, crs=5186)
    g.to_file(tmp_path / "contour.gpkg")
    gpd.GeoDataFrame({"CONT": [25.0]}, geometry=[Point(150, 100)], crs=5186).to_file(tmp_path / "spot.gpkg")
    monkeypatch.setattr(settings, "dem_dir", str(tmp_path))
    monkeypatch.setattr("src.static_data.dem.DEM_CLIP", tmp_path / "out.tif")
    dem = Dem.build_clip((0, 0, 200, 200), "EPSG:5186", margin_m=0)
    assert dem.res == 5
    v = dem.sample(dem.arr, np.array([50.0, 100.0, 150.0]), np.array([100.0, 100.0, 100.0]))
    assert v[0] == pytest.approx(15, abs=0.5)
    assert v[1] == pytest.approx(20, abs=0.5)
    assert v[2] == pytest.approx(25, abs=0.5)   # 표고점 값 재현


def test_relink_moves_rows_to_new_bid(tmp_path, monkeypatch):
    from src.alerts import subscriptions as s
    monkeypatch.setattr(s, "DB", tmp_path / "t.db")
    s.init()
    s.relink(["A", "B", "C"])
    s.subscribe(1, "x@example.com")            # 건물 B
    s.add_opening(2, {"kind": "door", "sill_above_ground_m": 0})   # 건물 C
    stats = s.relink(["C", "X", "B"])          # 목록이 바뀜: B는 2번, C는 0번, A는 사라짐
    assert stats["moved"] == 2
    with s.conn() as c:
        assert c.execute("SELECT bid FROM subscriptions").fetchone()[0] == 2
        assert c.execute("SELECT bid FROM openings").fetchone()[0] == 0


def test_merge_sources_prefers_vworld_and_fills_from_shp():
    import geopandas as gpd
    from shapely.geometry import Point
    from src.static_data.build_static import merge_sources
    geo = [Point(0, 0).buffer(1)] * 3
    v = gpd.GeoDataFrame({"src_objectid": [2, 1], "gid": ["B", "A"], "floors_up": [4, 2], "is_banjiha": [True, False]},
                         geometry=geo[:2], crs=5186)
    s = gpd.GeoDataFrame({"src_objectid": [1, 2, 3], "gid": ["A", "B", "C"], "floors_up": [2, 3, 1],
                          "is_banjiha": [False, True, True], "dong": ["x"] * 3}, geometry=geo, crs=5186)
    m = merge_sources(v, s)
    assert list(m.gid) == ["A", "B", "C"]                  # 원천 객체번호 순
    assert list(m.src) == ["vworld", "vworld", "shp"]      # 브이월드에 없는 C만 SHP에서 보충
    assert m.loc[m.gid == "B", "floors_up"].item() == 4    # 공통 건물은 브이월드 값
    assert list(m.bid) == [0, 1, 2] and not m.gid.duplicated().any()
