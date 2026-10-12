"""호우 사건 시뮬레이터: 과거 호우마다 10분 간격으로 위험 모델을 돌려 건물별 '사건 요약'을 만든다.

- 강수: 기상청 AWS 410(기상청)·417(금천)·509(관악) 10분 강수를 건물마다 역거리가중(IDW, p=2)
- 하수관: 서울시 하수관로 수위 실측(2020년 이후만 있음), 건물에서 1km 안 최근접 수위계
- 개구부: 입력된 건물은 그 값, 나머지는 입력값 평균 (실시간 모델과 같은 규칙)
- 수식은 src/engine/risk_model.py 함수를 그대로 쓰고, 계수만 params로 바꿔 넣는다 (칼리브레이션 탐색용)
사건 요약(건물별): 최대 점수·등급·지표면 수심·개구부 초과수심, 최대 요소 점수, 처음 '경고' 이상이 된 시각
"""
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyproj

from src.calibration.kma_aws import STATIONS, load_day
from src.config import DATA
from src.engine.risk_model import (classify, excess_over, opening_score, rain_score, sewer_score, surface_depth,
                                   terrain_score, weighted_score)

_TO5186 = pyproj.Transformer.from_crs(4326, 5186, always_xy=True)
STN_LONLAT = {410: (126.91746, 37.4933), 417: (126.90016, 37.46551), 509: (126.95015, 37.45284)}
SEWER_DIR = DATA / "raw" / "sewer"


@dataclass
class Event:
    eid: str
    days: list            # YYYYMMDD
    flood: bool           # 침수흔적도에 신림동 침수가 있는 사건


def group_events(days: list[str], flood_days: dict[str, list[str]]) -> list[Event]:
    """연속된 호우일을 한 사건으로 묶는다. 침수 사건은 침수흔적도의 날짜 범위를 쓴다."""
    covered, events = set(), []
    for eid, ds in flood_days.items():
        events.append(Event(eid, sorted(ds), True))
        covered.update(ds)
    rest = sorted(d for d in days if d not in covered)
    cur = []
    for d in rest:
        if cur and (datetime.strptime(d, "%Y%m%d") - datetime.strptime(cur[-1], "%Y%m%d")).days == 1:
            cur.append(d)
        else:
            if cur:
                events.append(Event(f"{cur[0][:4]}-{cur[0][4:6]}-{cur[0][6:]}", cur, False))
            cur = [d]
    if cur:
        events.append(Event(f"{cur[0][:4]}-{cur[0][4:6]}-{cur[0][6:]}", cur, False))
    return sorted(events, key=lambda e: e.days[0])


def station_rain(event: Event) -> tuple[list[datetime], np.ndarray]:
    """[시각], [관측소, 시각] 10분 강수. 결측 칸은 NaN."""
    rows = [r for d in event.days for r in load_day(d)]
    times = sorted({t for _, t, _ in rows})
    idx = {t: i for i, t in enumerate(times)}
    stns = list(STATIONS)
    a = np.full((len(stns), len(times)), np.nan)
    for s, t, v in rows:
        a[stns.index(s), idx[t]] = v
    return times, a


def idw_weights(bx, by):
    sx, sy = _TO5186.transform([STN_LONLAT[s][0] for s in STATIONS], [STN_LONLAT[s][1] for s in STATIONS])
    d = np.hypot(bx[:, None] - np.array(sx)[None, :], by[:, None] - np.array(sy)[None, :])
    w = 1 / np.maximum(d, 50) ** 2
    return w / w.sum(axis=1, keepdims=True)


def window_sum(a: np.ndarray, k: int) -> np.ndarray:
    """[관측소, 시각] 10분 강수 → 최근 k칸 합 (NaN은 0으로 보되 칸 전부 NaN이면 NaN)."""
    z = np.nan_to_num(a)
    c = np.cumsum(np.pad(z, ((0, 0), (k, 0))), axis=1)
    s = c[:, k:] - c[:, :-k]
    cnt = np.cumsum(np.pad((~np.isnan(a)).astype(int), ((0, 0), (k, 0))), axis=1)
    n = cnt[:, k:] - cnt[:, :-k]
    return np.where(n > 0, s, np.nan)


def load_sewer(event: Event, stations: list[dict]):
    """수위계별 10분 간격 [시각] 수위·30분 상승속도. 파일이 없으면 None (2020년 이전 사건)."""
    obs = {}
    for d in event.days:
        p = SEWER_DIR / f"{d}.csv"
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                obs.setdefault(r["station"], []).append((datetime.fromisoformat(r["ts"]), float(r["level_m"])))
    return obs


def sewer_at(obs_s, times):
    """최근 30분 안 마지막 수위와 30분 전 수위 → (H0, rise30) 배열."""
    if not obs_s:
        return np.full(len(times), np.nan), np.full(len(times), np.nan)
    ts = np.array([t.timestamp() for t, _ in obs_s])
    lv = np.array([v for _, v in obs_s])
    order = np.argsort(ts)
    ts, lv = ts[order], lv[order]
    h0, r30 = [], []
    for t in times:
        x = t.timestamp()
        i = np.searchsorted(ts, x, side="right") - 1
        if i < 0 or x - ts[i] > 1800:
            h0.append(np.nan), r30.append(np.nan)
            continue
        j = np.searchsorted(ts, ts[i] - 1800, side="left")
        h0.append(lv[i])
        r30.append((lv[i] - lv[j]) / max((ts[i] - ts[j]) / 60, 1) if j < i else np.nan)
    return np.array(h0), np.array(r30)


def simulate(event: Event, static: pd.DataFrame, rm: dict, sill: np.ndarray, barrier: np.ndarray,
             st_idx: np.ndarray, stations: list[dict], h_crit: np.ndarray, rain_w: np.ndarray) -> pd.DataFrame:
    """한 사건 × 건물 → 사건 요약. rm = settings.yaml risk_model 구조(계수만 바꿔 탐색)."""
    times, sr = station_rain(event)
    p10, p30, p60 = window_sum(sr, 1), window_sum(sr, 3), window_sum(sr, 6)
    # 건물별 IDW (관측소 결측은 가중치 재정규화)
    def idw(x):
        ok = ~np.isnan(x)
        num = rain_w @ np.nan_to_num(x)
        den = rain_w @ ok.astype(float)
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    P30, P60 = idw(p30), idw(p60)            # [건물, 시각]
    I30 = P30 * 2
    obs = load_sewer(event, stations)
    n, T = len(static), len(times)
    util = np.full((n, T), np.nan)
    rise = np.full((n, T), np.nan)
    if obs is not None:
        for j, s in enumerate(stations):
            h0, r30 = sewer_at(obs.get(s["station"], []), times)
            m = st_idx == j
            util[m] = h0 / h_crit[j]
            rise[m] = r30
    sewer_ex = np.maximum(0, np.nan_to_num(util) - 1) * h_crit[np.maximum(st_idx, 0)][:, None]
    sewer_ex = np.where(np.isnan(util), 0, sewer_ex)
    lowland = np.nan_to_num(np.clip(-static["rel_elev_m"].to_numpy() / rm["surface"]["lowland_full_m"], 0, 1))
    depth = surface_depth(sewer_ex, np.nan_to_num(I30), lowland[:, None], rm["surface"])
    ex = excess_over(depth, sill[:, None])
    comps = {"opening": opening_score(ex), "sewer": sewer_score(util, rm["scores"]["sewer"]),
             "rain": rain_score(P60, rm["scores"]["rain60"]),
             "terrain": np.repeat(terrain_score(static["rel_elev_m"].to_numpy(), rm["scores"]["terrain"])[:, None], T, 1)}
    score = weighted_score(comps, rm["weights"])
    level = classify(score, ex, rm["levels"], rm["override"], rise, P60)
    first_high = np.where((level >= 2).any(axis=1), np.argmax(level >= 2, axis=1), -1)
    out = pd.DataFrame({
        "bid": static["bid"].to_numpy(), "event": event.eid,
        "max_score": np.nanmax(np.where(np.isnan(score), -1, score), axis=1),
        "max_level": level.max(axis=1), "max_depth": depth.max(axis=1), "max_excess": ex.max(axis=1),
        "max_s_opening": comps["opening"].max(axis=1), "max_s_rain": np.nanmax(np.nan_to_num(comps["rain"], nan=-1), axis=1),
        "max_s_sewer": np.where(np.isnan(comps["sewer"]).all(axis=1), np.nan, np.nanmax(np.nan_to_num(comps["sewer"], nan=-1), axis=1)),
        "s_terrain": comps["terrain"][:, 0], "max_P60": np.nanmax(np.nan_to_num(P60), axis=1),
        "max_I30": np.nanmax(np.nan_to_num(I30), axis=1), "max_util": np.where(np.isnan(util).all(axis=1), np.nan, np.nanmax(np.nan_to_num(util, nan=-1), axis=1)),
        "first_high": [times[i].isoformat(timespec="minutes") if i >= 0 else None for i in first_high],
        "sewer_available": obs is not None})
    return out
