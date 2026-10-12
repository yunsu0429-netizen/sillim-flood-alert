"""정적 데이터 생성: 신림동 건물 + 지형 피처 + 취약도 → data/processed/

실행: .venv\\Scripts\\python -m src.static_data.build_static [--buildings shp|vworld] [--dem-dir 폴더]
  --buildings vworld : 브이월드 API로 최신 건물정보를 받아 갱신 (VWORLD_API_KEY 필요)
  --dem-dir 폴더     : 5m DEM 도엽 또는 수치지형도 등고선 폴더 (.env DEM_DIR과 같음)
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import MultiPolygon

from src.config import MANUAL, PROCESSED, load_params, settings
from src.engine.rules import vulnerability
from src.static_data.buildings import load_sillim_buildings
from src.static_data.dem import Dem

log = logging.getLogger(__name__)
REF_GL = {"study1 (Revit 지오코딩 + 동일 DEM)": 38.505, "빔방울 매스 뷰어": 40.4}


def _exterior(geom):
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    return [[round(x, 6), round(y, 6)] for x, y in geom.exterior.coords]


def _optional_layer(name):
    path = MANUAL / name
    if not path.exists():
        log.warning("%s 없음 → 해당 취약도 항목 제외 후 가중치 재정규화", path.name)
        return None
    return gpd.read_file(path).to_crs(5186)


def _previous():
    """이전 결과(비교용). 없으면 None."""
    path = PROCESSED / "buildings_static.parquet"
    return gpd.read_parquet(path) if path.exists() else None


COMPARE_COLS = ["use_code", "floors_up", "floors_down", "height_raw"]


def merge_sources(v: gpd.GeoDataFrame, s: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """브이월드(v) 전체 + 브이월드에 없는 SHP(s) 건물. 같은 건물(gid)은 브이월드 값을 쓴다.
    원천 객체번호(src_objectid) 순으로 정렬한 뒤 bid를 다시 매긴다 (구독·개구부는 gid로 재연결)."""
    v = v.assign(src="vworld")
    extra = s[~s["gid"].isin(set(v["gid"]))].assign(src="shp")
    cols = [c for c in v.columns if c in extra.columns]
    merged = gpd.GeoDataFrame(pd.concat([v[cols], extra[cols]], ignore_index=True), crs=v.crs)
    merged["src_objectid"] = pd.to_numeric(merged["src_objectid"], errors="coerce")
    merged = merged.sort_values(["src_objectid", "gid"], kind="stable").reset_index(drop=True)
    merged["bid"] = merged.index.astype(int)
    return merged


def _log_changes(v, s):
    """같은 건물인데 두 출처 값이 다른 항목 수."""
    m = s.merge(v, on="gid", suffixes=("_shp", "_vw"))
    diffs = {}
    for c in COMPARE_COLS:
        a, b = m[f"{c}_shp"], m[f"{c}_vw"]
        if c == "height_raw":
            diffs[c] = int(((a.fillna(0) - b.fillna(0)).abs() > 0.01).sum())
        else:
            diffs[c] = int(((a.astype(str) != b.astype(str)) & ~(a.isna() & b.isna())).sum())
    log.info("공통 건물 %d동 중 브이월드 값이 SHP와 다른 수(브이월드 값 사용): %s", len(m), diffs)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", choices=["merged", "vworld", "shp"], default=settings.building_source)
    ap.add_argument("--dem-dir", default=None)
    args = ap.parse_args()
    if args.dem_dir is not None:
        settings.dem_dir = args.dem_dir
    p = load_params()
    prev = _previous()
    today = datetime.now().strftime("%Y-%m-%d")
    shp_name = Path(settings.building_shp).stem
    shp_date = f"{shp_name[-8:-4]}-{shp_name[-4:-2]}-{shp_name[-2:]}" if shp_name[-8:].isdigit() else shp_name
    labels = {"vworld": f"브이월드 {today} 수집", "shp": f"SHP {shp_date}"}
    if args.buildings in ("vworld", "merged"):
        # 브이월드가 실패하면 예외로 중단한다 (SHP로 조용히 대체하지 않음 — 필요하면 --buildings shp)
        from src.static_data.vworld import load_sillim_buildings_vworld
        v = load_sillim_buildings_vworld()
    if args.buildings == "merged":
        s = load_sillim_buildings()
        _log_changes(v, s)
        gdf = merge_sources(v, s)
        n_shp = int((gdf["src"] == "shp").sum())
        labels["shp"] += " (브이월드에 없음)"
        log.info("합친 결과: 브이월드 %d동 + SHP 보충 %d동 = %d동 (반지하 후보 %d + %d)",
                 len(gdf) - n_shp, n_shp, len(gdf), int(gdf[gdf.src == "vworld"].is_banjiha.sum()),
                 int(gdf[gdf.src == "shp"].is_banjiha.sum()))
        bsource = f"국토교통부 GIS건물통합정보: {labels['vworld']} + SHP({shp_date}) 보충 {n_shp}동"
    elif args.buildings == "vworld":
        gdf = v.assign(src="vworld")
        bsource = f"국토교통부 GIS건물통합정보: {labels['vworld']}"
    else:
        gdf = load_sillim_buildings().assign(src="shp")
        bsource = f"국토교통부 GIS건물통합정보 {Path(settings.building_shp).name}"
    gdf["src_label"] = gdf["src"].map(labels)
    dup = int(gdf["gid"].duplicated().sum())
    if dup:
        log.warning("건물 고유 ID(gid) 중복 %d건 — 구독 재연결이 모호할 수 있음", dup)

    cent = gdf.geometry.representative_point()
    dem = Dem.build_clip(tuple(gdf.total_bounds), "EPSG:5186")
    to_dem = pyproj.Transformer.from_crs(5186, dem.crs, always_xy=True)
    cx, cy = to_dem.transform(cent.x.values, cent.y.values)
    radius = p["vulnerability"]["lowland_radius_m"]
    gl = dem.sample(dem.arr, cx, cy)
    mean_r = dem.sample(dem.focal_mean(radius), cx, cy)
    slope = dem.sample(dem.slope_pct(), cx, cy)
    gdf["gl_m"] = gl
    gdf["rel_elev_m"] = gl - mean_r
    gdf["slope_pct"] = slope
    log.info("지반고 결측 %d동 (DEM 범위 밖)", int(np.isnan(gl).sum()))

    streams = _optional_layer("streams.geojson")
    gdf["stream_dist_m"] = cent.distance(streams.union_all()) if streams is not None else np.nan
    history = _optional_layer("flood_history.geojson")
    gdf["flood_count"] = (cent.apply(lambda pt: int(history.contains(pt).sum()))
                          if history is not None else np.nan)

    def _v(r):
        return vulnerability(p, r.is_banjiha, r.rel_elev_m, r.slope_pct,
                             None if np.isnan(r.stream_dist_m) else r.stream_dist_m,
                             None if np.isnan(r.flood_count) else r.flood_count)
    gdf["vuln"] = [_v(r) for r in gdf.itertuples()]

    fh = p["engine"]["floor_height_m"]
    gdf["height_m"] = np.where(gdf["height_raw"].fillna(0) > 0, gdf["height_raw"].fillna(0),
                               np.maximum(gdf["floors_up"], 1) * fh)
    gdf["below_m"] = gdf["floors_down"] * fh

    ref = gdf[gdf["jibun"] == "419-1"]
    if len(ref):
        r = ref.iloc[0]
        old = prev[prev.jibun == "419-1"].gl_m.iloc[0] if prev is not None and (prev.jibun == "419-1").any() else None
        log.info("신림동 419-1 지반고 비교: 이번 DEM(%.0f m 격자)=%.3f m | 이전 결과=%s | %s", dem.res, r.gl_m,
                 f"{old:.3f} m" if old is not None else "-", " | ".join(f"{k}={v} m" for k, v in REF_GL.items()))
    b = gdf[gdf.is_banjiha]
    log.info("반지하 취약도 V: 서로 다른 값 %d개 (이전 %s개), 분위수 10/50/90%% = %s", b.vuln.round(2).nunique(),
             prev[prev.is_banjiha].vuln.round(2).nunique() if prev is not None else "-",
             np.nanpercentile(b.vuln, [10, 50, 90]).round(2).tolist())
    if prev is not None and "gid" in prev.columns:
        new_ids, old_ids = set(gdf.gid), set(prev.gid)
        log.info("건물 변화: 새로 생김 %d동, 사라짐 %d동 (이전 %d동 → 이번 %d동)",
                 len(new_ids - old_ids), len(old_ids - new_ids), len(prev), len(gdf))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    cols = ["bid", "gid", "src", "src_label", "pnu", "jibun", "use_name", "structure", "approved", "floors_up", "floors_down",
            "height_m", "below_m", "is_banjiha", "gl_m", "rel_elev_m", "slope_pct", "stream_dist_m",
            "flood_count", "vuln", "geometry"]
    gdf[cols].to_parquet(PROCESSED / "buildings_static.parquet")

    g4326 = gdf.to_crs(4326)
    feats = []
    for r, geom in zip(gdf.itertuples(), g4326.geometry):
        feats.append({"id": r.bid, "j": r.jibun if isinstance(r.jibun, str) else None, "u": r.use_name if isinstance(r.use_name, str) else None, "fu": r.floors_up, "fd": r.floors_down,
                      "h": round(float(r.height_m), 1), "bd": round(float(r.below_m), 1),
                      "z": None if np.isnan(r.gl_m) else round(float(r.gl_m), 2),
                      "b": bool(r.is_banjiha), "p": _exterior(geom)})
    meta = {"region": p["region"]["name"], "count": len(feats), "banjiha": int(gdf.is_banjiha.sum()),
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "sources": {"buildings": bsource,
                        "dem": (f"국토지리정보원 5m 지형 ({settings.dem_dir}, {dem.crs}, {dem.res:.0f} m)" if settings.dem_dir
                                else f"국토지리정보원 공개 DEM 37612 ({dem.crs}, {dem.res:.0f} m)")},
            "bounds": [float(v) for v in g4326.total_bounds]}
    with open(PROCESSED / "buildings_3d.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "buildings": feats}, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    log.info("저장 완료: 건물 %d동 (반지하 후보 %d동), bounds=%s", meta["count"], meta["banjiha"], meta["bounds"])


if __name__ == "__main__":
    main()
