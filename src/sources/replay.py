"""재생 모드 강우 (2022-08-08).

기본: 기상청 AWS 10분 실측(410 기상청·417 금천·509 관악, data/raw/rainfall/10min/)을 신림동 중심으로 IDW(p=2) 합성.
      SWMM 선행 적분을 위해 12:00부터, 화면 재생은 하수관 실측이 있는 18:00부터.
대체: 실측 파일이 없으면 study1 SWMM 입력의 Aug8_2022 하이토그래프 — 실측이 아니라 Huff 3분위 분포를 보도된
      첨두강도 130mm/hr에 맞춰 스케일한 "가정"값 (SWMM INTENSITY 형식, 5분 간격 mm/hr).
"""
import re
from datetime import datetime, timedelta

from src.config import settings

LABEL_SYNTH = "모의 데이터: 2022-08-08 관악 호우 재현 하이토그래프 (Huff 3분위, 첨두 130mm/hr 가정)"
LABEL_OBS = "실측: 2022-08-08 10분 강우 (기상청 AWS + 서울시 강우량계, 신림동 거리가중)"
LABEL = LABEL_OBS
SERIES_FROM = datetime(2022, 8, 8, 12, 0)     # 선행 강우(SWMM 6시간 적분)
REPLAY_FROM = datetime(2022, 8, 8, 18, 0)     # 화면 재생 시작 (하수관 실측 시작)
REPLAY_TO = datetime(2022, 8, 9, 2, 0)
ROW = re.compile(r"^Aug8_2022\s+(\d\d/\d\d/\d{4})\s+(\d\d:\d\d)\s+([\d.]+)\s*$", re.MULTILINE)


def load_intensity(path: str | None = None) -> list[tuple[datetime, float]]:
    with open(path or settings.replay_inp, encoding="utf-8") as f:
        text = f.read()
    rows = [(datetime.strptime(f"{d} {t}", "%m/%d/%Y %H:%M"), float(v)) for d, t, v in ROW.findall(text)]
    if not rows:
        raise ValueError("Aug8_2022 TIMESERIES를 찾지 못함")
    return rows


def to_10min_depths(rows: list[tuple[datetime, float]], step_min: int = 10) -> list[tuple[datetime, float]]:
    """(구간 끝 시각, 10분 강우량 mm) 리스트. 강도는 다음 정의점까지 계단식으로 유지."""
    start, end = rows[0][0], rows[-1][0]
    out = []
    t = start
    idx = 0
    while t < end:
        t_end = t + timedelta(minutes=step_min)
        depth = 0.0
        m = t
        while m < t_end:
            while idx + 1 < len(rows) and rows[idx + 1][0] <= m:
                idx += 1
            depth += rows[idx][1] / 60.0  # 1분 단위 적분
            m += timedelta(minutes=1)
        out.append((t_end, round(depth, 2)))
        t = t_end
    return out


def load_observed() -> list[tuple[datetime, float]] | None:
    """10분 실측 → 신림동 반지하 후보 중심 IDW(p=2) (구간 끝 시각, mm). 파일이 없으면 None.
    관측망: 기상청 AWS 3곳(410·417·509) + 서울시 강우량계 전체(OA-22715, 10min_seoul_gauge) — v0.3 보정과 같은 관측망."""
    import numpy as np
    import pandas as pd
    import pyproj
    from src.calibration.simulate import STN_LONLAT
    from src.config import DATA, PROCESSED
    days = ("20220808", "20220809")
    paths = [DATA / "raw" / "rainfall" / "10min" / f"{d}.csv" for d in days]
    if not all(p.exists() for p in paths):
        return None
    r = pd.concat([pd.read_csv(p) for p in paths])
    lonlat = dict(STN_LONLAT)
    gdir = DATA / "raw" / "rainfall" / "10min_seoul_gauge"
    if all((gdir / f"{d}.csv").exists() for d in days) and (gdir / "stations.csv").exists():
        st = pd.read_csv(gdir / "stations.csv")
        lonlat.update({int(a): (o, t) for a, o, t in zip(st.stn, st.lon, st.lat)})
        r = pd.concat([r, *[pd.read_csv(gdir / f"{d}.csv") for d in days]])
    r["ts"] = pd.to_datetime(r.ts)
    r = r[(r.ts > SERIES_FROM) & (r.ts <= REPLAY_TO) & r.stn.isin(lonlat)]
    a = r.pivot_table(index="ts", columns="stn", values="rn10")
    import geopandas as gpd
    b = gpd.read_parquet(PROCESSED / "buildings_static.parquet", columns=["is_banjiha", "geometry"])
    c = b[b.is_banjiha].geometry.representative_point()
    tr = pyproj.Transformer.from_crs(4326, 5186, always_xy=True)
    sx, sy = tr.transform([lonlat[s][0] for s in a.columns], [lonlat[s][1] for s in a.columns])
    w = 1 / np.maximum(np.hypot(np.array(sx) - c.x.mean(), np.array(sy) - c.y.mean()), 50) ** 2
    ok = a.notna().to_numpy()
    v = (np.nan_to_num(a.to_numpy()) @ w) / np.maximum(ok @ w, 1e-9)
    return [(t.to_pydatetime(), round(float(x), 2)) for t, x in zip(a.index, v)]


def load_series() -> list[tuple[datetime, float]]:
    """실측 우선, 없으면 가정 하이토그래프. (LABEL도 함께 맞춘다)"""
    global LABEL
    obs = load_observed()
    if obs:
        LABEL = LABEL_OBS
        return obs
    LABEL = LABEL_SYNTH
    return to_10min_depths(load_intensity())


GAUGE_STN = {22301: "관악구청", 22302: "신림P"}       # 서울시 강우량계 코드 (OA-22715, src/calibration/seoul_gauges.py)


def load_gauges() -> dict[str, list[tuple[datetime, float]]] | None:
    """관악구 서울시 강우량계 2곳의 2022-08-08 10분 실측 (자치구 침수예보 기준 판정용). 파일 없으면 None."""
    import pandas as pd
    from src.config import DATA
    paths = [DATA / "raw" / "rainfall" / "10min_seoul_gauge" / f"{d}.csv" for d in ("20220808", "20220809")]
    if not all(p.exists() for p in paths):
        return None
    r = pd.concat([pd.read_csv(p) for p in paths])
    r["ts"] = pd.to_datetime(r.ts)
    r = r[r.stn.isin(GAUGE_STN) & (r.ts > SERIES_FROM - timedelta(hours=3)) & (r.ts <= REPLAY_TO)]
    return {GAUGE_STN[s]: [(t.to_pydatetime(), float(v)) for t, v in zip(g.ts, g.rn10)] for s, g in r.groupby("stn")}
