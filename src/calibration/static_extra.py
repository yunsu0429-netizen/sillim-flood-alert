"""서울시 공공데이터로 건물 정적 특성 추가 (개선 D의 정적 취약성 모형 입력).

  fmap_depth_m   서울시 풍수해 침수예상도(OA-21172, EPSG:5186) 건물 대표점이 속한 침수심 구간의 대표값
                 (1: 0~0.5m→0.25, 2: 0.5~1.0→0.75, 3: 1.0~1.5→1.25, 4: 1.5~2.0→1.75, 5: 2.0~3.0→2.5, 6: ≥3.0→3.0, 밖 0)
  fmap_dist_m    가장 가까운 침수예상 구역까지 거리 (안이면 0)
  pump_dist_m    가장 가까운 빗물펌프장(OA-21179, 118곳)까지 거리
  gu_imperv_pct  자치구 불투수면 비율(OA-22363, 2020 서울도시생태현황도)
  sgg_name       자치구 이름 (서울시 강우량계 구청명과 연결용)

실행: .venv\\Scripts\\python -m src.calibration.static_extra
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src.calibration import seoul_data as sd
from src.calibration import seoul_gauges

log = logging.getLogger(__name__)
SRC = sd.DATA / "raw" / "seoul_open"
OUT = sd.OUT / "static_extra.parquet"
FMAP_DEPTH = {1: 0.25, 2: 0.75, 3: 1.25, 4: 1.75, 5: 2.5, 6: 3.0}
FEATS = ["fmap_depth_m", "fmap_dist_m", "pump_dist_m", "gu_imperv_pct"]


def sgg_names(b: pd.DataFrame) -> dict:
    """시군구코드 → 자치구 이름: 서울시 강우량계(구청명, 위경도)에서 가장 가까운 건물의 시군구코드."""
    import pyproj
    from scipy.spatial import cKDTree
    st = seoul_gauges.stations()
    tr = pyproj.Transformer.from_crs(4326, 5186, always_xy=True)
    gx, gy = tr.transform(st.lon.to_numpy(), st.lat.to_numpy())
    _, j = cKDTree(np.c_[b.x, b.y]).query(np.c_[gx, gy])
    m = pd.DataFrame({"sgg": b.sgg.to_numpy()[j], "gu": st.gu.to_numpy()})
    return m.groupby("sgg").gu.agg(lambda s: s.mode().iloc[0]).to_dict()


def build() -> pd.DataFrame:
    b = pd.read_parquet(sd.OUT / "buildings.parquet", columns=["bid", "sgg", "x", "y"])
    pts = shapely.points(b.x.to_numpy(), b.y.to_numpy())
    depth = np.zeros(len(b))
    parts = []
    for k, v in FMAP_DEPTH.items():
        g = gpd.read_file(SRC / "flood_map" / f"DS_FLOODING_{k}.shp", encoding="cp949")
        geoms = shapely.get_parts(shapely.make_valid(g.geometry.values))      # 원본 폴리곤에 자기교차가 있어 개별 보정
        geoms = geoms[shapely.get_type_id(geoms) == 3]                         # Polygon만
        parts.append(geoms)
        tree = shapely.STRtree(geoms)
        pi, _ = tree.query(pts, predicate="within")
        hit = np.zeros(len(b), dtype=bool)
        hit[np.unique(pi)] = True
        depth = np.where(hit, np.maximum(depth, v), depth)
    tree = shapely.STRtree(np.concatenate(parts))
    _, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    pump = gpd.read_file(SRC / "rainpump" / "RAINPUMP.shp", encoding="cp949")
    pxy = shapely.get_coordinates(pump.geometry.values)
    from scipy.spatial import cKDTree
    pd_, _ = cKDTree(pxy).query(np.c_[b.x, b.y])
    names = sgg_names(b)
    imp = pd.read_csv(SRC / "imperv.csv", encoding="cp949")
    imp.columns = [c.strip() for c in imp.columns]
    imp_map = dict(zip(imp["자치구"].str.strip(), pd.to_numeric(imp["불투수면적 비율(퍼센트)"], errors="coerce")))
    gu = b.sgg.map(names)
    out = pd.DataFrame({"bid": b.bid, "fmap_depth_m": depth, "fmap_dist_m": np.where(depth > 0, 0.0, dist),
                        "pump_dist_m": pd_, "gu_imperv_pct": gu.map(imp_map).to_numpy(), "sgg_name": gu.to_numpy()})
    out.to_parquet(OUT)
    log.info("침수예상도 안 %.1f%%, 펌프장 거리 중앙값 %.0fm, 구 이름 연결 %d/%d", 100 * (depth > 0).mean(),
             np.median(pd_), out.sgg_name.notna().sum(), len(out))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()
