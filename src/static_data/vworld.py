"""브이월드 국가공간정보(NED) WFS로 신림동 GIS건물통합정보를 받아 SHP 로더와 같은 형태로 돌려준다.

확인 결과 (2026-10-07, 신림동 419-1 부근 응답):
- 일반 WFS(/req/wfs)의 건물 레이어는 도로명주소건물(lt_c_spbd)뿐이라 주용도·높이가 없다.
- GIS건물통합정보는 /ned/wfs/getBldgisSpceWFS, typename=dt_d010 으로 제공되며 SHP(AL_D010)와 같은 값이다.
  gis_idntfc_no=A1, pnu=A2, ld_cpsg_code=A3, mnnm·slno=지번, regstr_se_code=A6(2=산), buld_prpos_code=A8,
  strct_code=A10, use_confm_de=A13, hg=A16, ground_floor_co=A26, undgrnd_floor_co=A27
- 주용도·구조는 코드만 오므로 이름은 SHP에서 뽑은 대응표(USE_NAMES, STRUCT_NAMES)로 붙인다.

확인용: .venv\\Scripts\\python -m src.static_data.vworld --probe
요청당 최대 객체 수 제한이 있어 신림동 범위를 격자로 나눠 받고, 한 칸이 가득 차면 4등분해 다시 받는다.
"""
import argparse
import logging
import time
import xml.etree.ElementTree as ET

import geopandas as gpd
import httpx
import pandas as pd
from shapely.geometry import shape

from src.config import load_params, settings

log = logging.getLogger(__name__)
WFS = "https://api.vworld.kr/req/wfs"
NED_WFS = "https://api.vworld.kr/ned/wfs/getBldgisSpceWFS"
# 신림동 법정동 범위 (EPSG:5186, 건물통합정보 SHP 추출 결과 기준, 여유 200m)
SILLIM_BOX_5186 = (190939.0, 537407.0, 197534.0, 543657.0)
# AL_D010 SHP(신림동)의 A8→A9, A10→A11 실제 값 (2026-09-09 파일)
USE_NAMES = {"01000": "단독주택", "02000": "공동주택", "03000": "제1종근린생활시설", "04000": "제2종근린생활시설",
             "05000": "문화및집회시설", "06000": "종교시설", "07000": "판매시설", "09000": "의료시설",
             "10000": "교육연구시설", "11000": "노유자시설", "12000": "수련시설", "13000": "운동시설",
             "14000": "업무시설", "15000": "숙박시설", "16000": "위락시설", "17000": "공장", "18000": "창고시설",
             "19000": "위험물저장및처리시설", "20000": "자동차관련시설", "21000": "동.식물 관련시설",
             "24000": "방송통신시설", "Z3000": "근린생활시설", "Z8000": "교육연구및복지시설"}
STRUCT_NAMES = {"11": "벽돌구조", "12": "블록구조", "13": "석구조", "19": "기타조적구조", "21": "철근콘크리트구조",
                "29": "기타콘크리트구조", "31": "일반철골구조", "32": "경량철골구조", "33": "강파이프구조",
                "39": "기타강구조", "41": "철골콘크리트구조", "42": "철골철근콘크리트구조",
                "49": "기타철골철근콘크리트구조", "51": "일반목구조", "99": "기타구조"}


def _cfg() -> dict:
    v = load_params().get("vworld") or {}
    return {"typename": v.get("typename", "dt_d010"), "fields": v.get("fields") or {},
            "max_features": int(v.get("max_features", 1000)), "tile_m": float(v.get("tile_m", 500)),
            "domain": v.get("domain", "http://localhost:5173")}


def _auth() -> dict:
    if not settings.vworld_api_key:
        raise RuntimeError("VWORLD_API_KEY가 없습니다 (.env에 브이월드 인증키를 넣어야 함)")
    return {"key": settings.vworld_api_key, "domain": _cfg()["domain"]}


def list_layers() -> list[tuple[str, str]]:
    """일반 WFS GetCapabilities → (레이어명, 제목) 목록."""
    r = httpx.get(WFS, params=_auth() | {"SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetCapabilities"}, timeout=30)
    r.raise_for_status()
    out = []
    for ft in ET.fromstring(r.content).iter():
        if ft.tag.endswith("FeatureType"):
            name = next((c.text for c in ft if c.tag.endswith("Name")), "")
            title = next((c.text for c in ft if c.tag.endswith("Title")), "")
            out.append((name, title or ""))
    return out


def get_features(box: tuple, max_features: int) -> list[dict]:
    x0, y0, x1, y1 = box
    cfg = _cfg()
    params = _auth() | {"typename": cfg["typename"], "bbox": f"{x0},{y0},{x1},{y1},EPSG:5186",
                        "srsName": "EPSG:5186", "maxFeatures": max_features, "output": "application/json"}
    for attempt in range(4):   # 연결 끊김·5xx 같은 일시 오류는 잠시 쉬었다 다시 (그래도 안 되면 예외로 중단)
        try:
            r = httpx.get(NED_WFS, params=params, timeout=60)
            if r.status_code < 500:
                break
            err = f"HTTP {r.status_code}"
        except (httpx.TransportError, httpx.TimeoutException) as e:
            err = str(e)
        if attempt == 3:
            raise RuntimeError(f"브이월드 요청 4회 실패 ({box}): {err}")
        log.warning("브이월드 요청 실패(%s) — %d초 뒤 재시도", err, 2 ** (attempt + 1))
        time.sleep(2 ** (attempt + 1))
    time.sleep(0.1)   # 연속 요청 간격
    r.raise_for_status()
    try:
        data = r.json()
    except ValueError as e:
        raise RuntimeError(f"브이월드 응답이 JSON이 아님: {r.text[:300]}") from e
    if "features" not in data:
        raise RuntimeError(f"브이월드 오류 응답: {str(data)[:300]}")
    return data["features"]


def _fetch_box(box, cap, depth=0) -> list[dict]:
    feats = get_features(box, cap)
    if len(feats) < cap or depth >= 6:
        if len(feats) >= cap:
            log.warning("격자 %s가 %d단계 분할 후에도 가득 참 — 일부 누락 가능", box, depth)
        return feats
    x0, y0, x1, y1 = box   # 가득 찼으면 4등분해서 다시
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    out = []
    for b in [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]:
        out += _fetch_box(b, cap, depth + 1)
    return out


def _jibun(mnnm, slno) -> str | None:
    """'0419','0001' → '419-1' (SHP A5와 같은 표기, 산 지번도 '산' 없이)."""
    try:
        m, s = int(mnnm), int(slno or 0)
    except (TypeError, ValueError):
        return None
    return f"{m}-{s}" if s else str(m)


def to_frame(feats: list[dict]) -> gpd.GeoDataFrame:
    """WFS 객체 → SHP 로더와 같은 컬럼의 GeoDataFrame (신림동 필터 전)."""
    fields = _cfg()["fields"]
    rows, geoms = [], []
    for f in feats:
        props = f.get("properties") or {}
        row = {ours: props.get(theirs) for theirs, ours in fields.items()}
        row["jibun"] = _jibun(props.get("mnnm"), props.get("slno"))
        rows.append(row)
        geoms.append(shape(f["geometry"]) if f.get("geometry") else None)
    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=5186)
    gdf["use_code"] = gdf["use_code"].where(gdf["use_code"].notna(), None)
    gdf["use_name"] = gdf["use_code"].map(USE_NAMES)
    gdf["structure"] = gdf.pop("struct_code").map(STRUCT_NAMES) if "struct_code" in gdf else None
    for c in ("floors_up", "floors_down"):
        gdf[c] = pd.to_numeric(gdf[c], errors="coerce").fillna(0).astype(int)
    gdf["height_raw"] = pd.to_numeric(gdf["height_raw"], errors="coerce")
    return gdf


def load_sillim_buildings_vworld() -> gpd.GeoDataFrame:
    cfg = _cfg()
    need = {"gid", "pnu", "bjd_code", "use_code", "floors_up", "floors_down", "height_raw"}
    missing = need - set(cfg["fields"].values())
    if missing:
        raise RuntimeError(f"config/settings.yaml vworld.fields 매핑에 필요한 항목 없음: {sorted(missing)}")
    x0, y0, x1, y1 = SILLIM_BOX_5186
    step = cfg["tile_m"]
    feats, n_tiles = [], 0
    x = x0
    while x < x1:
        y = y0
        while y < y1:
            feats += _fetch_box((x, y, min(x + step, x1), min(y + step, y1)), cfg["max_features"])
            n_tiles += 1
            y += step
        x += step
    log.info("브이월드 %s: 격자 %d칸, 객체 %d개 수신", cfg["typename"], n_tiles, len(feats))
    p = load_params()
    gdf = to_frame(feats)
    gdf = gdf.drop_duplicates(subset="gid")                 # 격자 경계에 걸친 건물 중복 제거
    gdf = gdf[gdf["bjd_code"].astype(str) == p["region"]["bjd_code"]]   # 신림동만
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
    gdf["is_banjiha"] = gdf["use_code"].isin(p["region"]["residential_use_codes"]) & (gdf["floors_down"] >= 1)
    # SHP 추출과 같은 순서(원본 객체 번호)로 정렬해 두면 bid가 SHP 결과와 최대한 같아진다
    gdf = gdf.sort_values("src_objectid").reset_index(drop=True) if "src_objectid" in gdf else gdf.reset_index(drop=True)
    gdf["bid"] = gdf.index.astype(int)
    log.info("브이월드 신림동 건물 %d동, 반지하 후보 %d동", len(gdf), int(gdf.is_banjiha.sum()))
    return gdf


def probe():
    """레이어 목록과 건물통합정보 샘플 필드를 출력한다."""
    layers = list_layers()
    bld = [(n, t) for n, t in layers if "bld" in n.lower() or "건물" in t]
    print(f"일반 WFS 레이어 {len(layers)}개, 건물 관련: {bld}")
    cx, cy = 193443.0, 542075.0   # 신림동 419-1 부근
    feats = get_features((cx - 30, cy - 30, cx + 30, cy + 30), 5)
    print(f"\nGIS건물통합정보({_cfg()['typename']}) 샘플 {len(feats)}개:")
    for f in feats[:2]:
        for k, v in (f.get("properties") or {}).items():
            print(f"  {k} = {v}")
        print("  ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    if ap.parse_args().probe:
        probe()
