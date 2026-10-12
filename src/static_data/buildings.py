"""건물통합정보(AL_D010) SHP에서 신림동 건물을 추출한다.

필드 매핑은 신림동 419-1 표본으로 확인함 (2026-10, AL_D010_11_20260909):
  A0 원천 객체번호  A1 GIS건물통합식별번호(건물 고유 ID → gid)  A2 고유번호(PNU)  A3 법정동코드  A4 법정동명  A5 지번  A8 주용도코드  A9 주용도명
  A11 구조명  A12 건축면적(m2)  A13 사용승인일  A16 높이(m, 0=없음)  A26 지상층수  A27 지하층수
"""
import logging

import geopandas as gpd
import pyogrio

from src.config import load_params, settings

log = logging.getLogger(__name__)

FIELDS = {"A0": "src_objectid", "A1": "gid", "A2": "pnu", "A4": "dong", "A5": "jibun", "A8": "use_code", "A9": "use_name",
          "A11": "structure", "A12": "bldg_area_m2", "A13": "approved", "A16": "height_raw", "A26": "floors_up", "A27": "floors_down"}


def load_sillim_buildings() -> gpd.GeoDataFrame:
    params = load_params()
    code = params["region"]["bjd_code"]
    gdf = pyogrio.read_dataframe(settings.building_shp, where=f"A3='{code}'", encoding="cp949",
                                 columns=["A3", *FIELDS])
    log.info("SHP CRS=%s, 신림동 건물 %d동", gdf.crs, len(gdf))
    if gdf.crs is None or gdf.crs.to_epsg() != 5186:
        raise ValueError(f"예상한 CRS(EPSG:5186)가 아님: {gdf.crs}")
    gdf = gdf.drop(columns="A3").rename(columns=FIELDS)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
    gdf["floors_up"] = gdf["floors_up"].fillna(0).astype(int)
    gdf["floors_down"] = gdf["floors_down"].fillna(0).astype(int)
    residential = gdf["use_code"].isin(params["region"]["residential_use_codes"])
    gdf["is_banjiha"] = residential & (gdf["floors_down"] >= 1)
    gdf = gdf.reset_index(drop=True)
    gdf["bid"] = gdf.index.astype(int)
    log.info("반지하 후보(주거 & 지하층>=1): %d동", int(gdf["is_banjiha"].sum()))
    return gdf
