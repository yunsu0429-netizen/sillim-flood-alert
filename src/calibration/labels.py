"""침수 정답(label): 서울시 침수흔적도 × 신림동 건물 공간조인 (prototype.md §49, §53).

- 침수흔적도 SHP (2010~2025, EPSG:5179) → EPSG:5186, 신림동 건물 폴리곤과 교차
- 건물별 겹친 비율 = 교차 면적 / 건물 면적, 침수심 = 겹친 흔적 폴리곤의 F_SHIM 최댓값
- flood = 겹친 비율 ≥ min_ratio (경계에 살짝 걸친 건물 오분류 방지)
- 라벨 출처 구분: 조사(현장 조사 침수심) / 지원금·보험(신고 기반, 침수심 0.2 일괄)
- 호우 당시 없던 건물(사용승인일 > 호우일)은 그 사건에서 제외
"""
import glob
import logging
import warnings

import geopandas as gpd
import pandas as pd
import pyogrio

from src.config import DATA, PROCESSED

log = logging.getLogger(__name__)
TRACE_DIR = DATA / "raw" / "flood_trace"
# 신림동에 침수흔적이 있는 호우 (침수흔적도 F_DISA_NM / F_SAT_YMD 기준)
FLOOD_EVENTS = {
    "2010-09-21": {"year": "2010", "days": ["20100921"]},
    "2011-07-27": {"year": "2011", "days": ["20110726", "20110727", "20110728"]},
    "2022-08-08": {"year": "2022", "days": ["20220808", "20220809"]},
}


def load_trace(year: str) -> gpd.GeoDataFrame:
    files = glob.glob(str(TRACE_DIR / year / "*.shp"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pyogrio.read_dataframe(files[0], encoding="cp949")
    if d.crs is None:
        raise ValueError(f"{files[0]} 좌표계 없음")
    d = d.to_crs(5186)
    d = d[d.geometry.notna() & d.geometry.is_valid]
    nm = d["F_DISA_NM"].astype(str)
    d["label_kind"] = (nm.str.contains("지원금|보험")).map({True: "신고(지원금·보험)", False: "조사"})
    return d


def event_labels(event: str, buildings: gpd.GeoDataFrame, min_ratio: float = 0.1) -> pd.DataFrame:
    ev = FLOOD_EVENTS[event]
    tr = load_trace(ev["year"])
    tr = tr[tr["F_SAT_YMD"].astype(str).str[:8].isin(ev["days"])]
    hull = buildings.union_all().convex_hull
    tr = tr[tr.intersects(hull)]
    j = gpd.overlay(buildings[["bid", "geometry"]], tr[["F_SHIM", "label_kind", "geometry"]], how="intersection",
                    keep_geom_type=True)
    j["inter_area"] = j.geometry.area
    agg = j.groupby("bid").agg(inter_area=("inter_area", "sum"), depth=("F_SHIM", "max"),
                               survey=("label_kind", lambda s: (s == "조사").any()))
    out = pd.DataFrame({"bid": buildings["bid"], "event": event})
    out = out.merge(agg, on="bid", how="left")
    area = buildings.set_index("bid").geometry.area
    out["ratio"] = (out["inter_area"].fillna(0) / out["bid"].map(area)).clip(0, 1)
    out["flood"] = (out["ratio"] >= min_ratio).astype(int)
    out["depth"] = out["depth"].where(out["flood"] == 1)
    out["label_kind"] = out["survey"].map({True: "조사", False: "신고(지원금·보험)"}).where(out["flood"] == 1)
    log.info("%s 흔적 폴리곤 %d개 → 침수 건물 %d동 (반지하 후보 %d동)", event, len(tr), int(out.flood.sum()),
             int(out.flood[buildings.set_index("bid").loc[out.bid, "is_banjiha"].to_numpy()].sum()))
    return out[["bid", "event", "flood", "ratio", "depth", "label_kind"]]


def built_by(buildings, day: str):
    """사용승인일이 없거나 day 이전이면 True (그 호우 때 있던 건물)."""
    ap = pd.to_datetime(buildings["approved"], errors="coerce")
    return (ap.isna() | (ap <= pd.Timestamp(day))).to_numpy()


def build_all(buildings=None) -> pd.DataFrame:
    b = buildings if buildings is not None else gpd.read_parquet(PROCESSED / "buildings_static.parquet")
    return pd.concat([event_labels(e, b) for e in FLOOD_EVENTS], ignore_index=True)
