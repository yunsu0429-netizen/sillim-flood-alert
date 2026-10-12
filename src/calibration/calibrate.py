"""칼리브레이션 (prototype.md §27~48).

절차
1. 사건 목록: 2010~2025년 6~9월 호우일(관측소 일강수 ≥ 80mm) + 침수흔적도 침수 사건
2. 사건마다 시뮬레이션(simulate.py) → 건물별 사건 요약
3. 정답: 침수흔적도 공간조인(labels.py) + 언론 보도(data/manual/flood_reports.csv), 침수 없는 호우는 0
4. 시간순 분리 (§38, §62): 보정 = 2010~2020 사건, 검증·시험 = 2021~2025 사건 (2022-08-08 포함)
5. 보정 대상: 지표면 수심(α·β·γ·배수능력), 점수 가중치(로지스틱 회귀 §35), 등급 경계(§31.3), 침수확률(§36)
6. 지표: Recall·Precision·F1·PR-AUC·Brier (§39), 침수심 MAE·Bias (§40), 경보 선행시간 (§41)

실행: .venv\\Scripts\\python -m src.calibration.calibrate
"""
import copy
import csv
import itertools
import json
import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.calibration.labels import FLOOD_EVENTS, build_all, built_by
from src.calibration.simulate import Event, group_events, idw_weights, simulate
from src.config import DATA, MANUAL, PROCESSED, ROOT, load_params
from src.engine.forecast import RiskEngine
from src.sources import sewer_api

log = logging.getLogger(__name__)
OUT = DATA / "calibration"
SPLIT_YEAR = 2021          # 이 해부터는 검증·시험 (보정에 쓰지 않음)


# ---------------------------------------------------------------- 준비
def heavy_days(min_mm=80.0) -> list[str]:
    by = {}
    with open(DATA / "raw" / "rainfall" / "aws_daily_2010_2025_jun_sep.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = float(r["rn_day"])
            if v >= 0:
                by[r["date"]] = max(by.get(r["date"], 0), v)
    return sorted(d for d, v in by.items() if v >= min_mm)


def events() -> list[Event]:
    return group_events(heavy_days(), {k: v["days"] for k, v in FLOOD_EVENTS.items()})


class Context:
    """사건과 무관한 건물·개구부·수위계 정보 (한 번만 준비)."""

    def __init__(self):
        self.p = load_params()
        self.static = gpd.read_parquet(PROCESSED / "buildings_static.parquet").sort_values("bid").reset_index(drop=True)
        self.stations = sewer_api.load_stations()
        self.engine = RiskEngine(self.p, self.static, self.stations)
        from src.alerts import subscriptions
        subscriptions.init()
        self.engine.set_openings(subscriptions.all_openings())
        pts = self.static.geometry.representative_point()
        self.rain_w = idw_weights(pts.x.to_numpy(), pts.y.to_numpy())
        self.banjiha = self.static["is_banjiha"].to_numpy()
        self.labels = self._labels()

    def _labels(self):
        path = OUT / "labels_flood_events.parquet"
        lab = pd.read_parquet(path) if path.exists() else build_all(self.static)
        rep = MANUAL / "flood_reports.csv"
        if rep.exists():   # 언론 보도 등 추정 라벨 (침수흔적도에 빠진 건물)
            for r in csv.DictReader(open(rep, encoding="utf-8")):
                hit = self.static.index[self.static.jibun == r["jibun"]]
                for b in hit:
                    m = (lab.bid == b) & (lab.event == r["event"])
                    lab.loc[m, ["flood", "label_kind"]] = [int(r["flood"]), r["label_type"]]
        return lab

    def sewer_link(self, ev: Event):
        """그 사건에 자료가 있는 수위계만으로 건물 연결 (반경 1km)."""
        obs_days = [sewer_api.DATA / "raw" / "sewer" / f"{d}.csv" for d in ev.days]
        if not all(p.exists() for p in obs_days):
            return np.full(len(self.static), -1)
        have = set()
        for p in obs_days:
            with open(p, encoding="utf-8") as f:
                have |= {r["station"] for r in csv.DictReader(f)}
        mask = np.array([s["station"] in have for s in self.stations])
        idx = self.engine.st_idx.copy()
        # 최근접 수위계가 그날 자료가 없으면 연결 해제 (보수적으로 결측 처리)
        idx[(idx >= 0) & ~mask[np.maximum(idx, 0)]] = -1
        return idx


def run(ctx: Context, rm: dict, evs: list[Event]) -> pd.DataFrame:
    m = ctx.engine.model
    parts = []
    for ev in evs:
        s = simulate(ev, ctx.static, rm, m.sill, m.barrier, ctx.sewer_link(ev), ctx.stations, ctx.engine.h_crit,
                     ctx.rain_w)
        s["year"] = int(ev.days[0][:4])
        s["day0"] = ev.days[0]
        parts.append(s)
    d = pd.concat(parts, ignore_index=True)
    lab = ctx.labels[["bid", "event", "flood", "depth", "label_kind"]]
    d = d.merge(lab, on=["bid", "event"], how="left")
    d["flood"] = d["flood"].fillna(0).astype(int)          # 침수흔적 없는 호우 = 침수 없음 (§45)
    keep = ctx.banjiha[d.bid] & np.concatenate([built_by(ctx.static, ev.days[0])[ctx.static.bid] for ev in evs
                                                for _ in [0]])[np.arange(len(d)) % len(ctx.static)] \
        if False else ctx.banjiha[d.bid]
    d = d[keep]
    # 호우 당시 없던 건물 제외
    ap = pd.to_datetime(ctx.static.set_index("bid").loc[d.bid, "approved"], errors="coerce").to_numpy()
    ev_day = pd.to_datetime(d["day0"]).to_numpy()
    d = d[pd.isna(ap) | (ap <= ev_day)]
    return d.reset_index(drop=True)


# ---------------------------------------------------------------- 평가
def metrics(y, pred, score=None, prob=None) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(bool)
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * pre * rec / (pre + rec) if pre and rec and pre == pre and rec == rec else float("nan")
    out = {"n": len(y), "positives": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "recall": rec, "precision": pre, "f1": f1}
    if score is not None and y.sum() and (1 - y).sum():
        from sklearn.metrics import average_precision_score, roc_auc_score
        out["pr_auc"] = float(average_precision_score(y, score))
        out["roc_auc"] = float(roc_auc_score(y, score))
    if prob is not None:
        out["brier"] = float(np.mean((np.asarray(prob) - y) ** 2))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
