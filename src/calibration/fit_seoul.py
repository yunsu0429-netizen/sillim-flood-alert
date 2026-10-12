"""서울 전역 확장 보정 시험 (fit.py와 같은 단계, 표본만 서울 전역으로).

표본
- 건물: 서울 반지하 후보 27.5만 동 중, 어느 사건에서든 침수된 건물 전부 + 나머지에서 무작위 NEG_SAMPLE동.
  무작위 표본은 가중치(전체 비침수 수 ÷ 표본 수)를 주어 정밀도·PR-AUC가 전체 기준이 되게 한다.
- 사건: seoul_data.events (호우일 + 침수흔적). 늦은 신고와만 겹친 건물(flood=-1)은 그 사건에서 제외.
- 강우: 서울 일대 AWS 61곳 10분 강수, 건물마다 IDW(p=2, 그날 자료 있는 관측소만).
- 하수관: 서울 수위계 245곳(지오코딩), 1km 안 최근접. 만관 수위 = 보정 기간(≤2022) 최고수위.
- 지형: 공개DEM 90m 4도엽 병합(서울 전역) → 상대고도(주변 300m 평균 대비)·경사.
- 개구부: 실측이 없어 전부 신림동 입력 평균(0.10m).
시간 분리: 보정 = 2010~2022 사건, 시험 = 2023~2025 사건.

실행: .venv\\Scripts\\python -m src.calibration.fit_seoul
"""
import copy
import itertools
import json
import logging
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import pyproj

from src.calibration import seoul_data as sd
from src.calibration.fit import COMPS, evaluate
from src.calibration.simulate import sewer_at, window_sum
from src.config import load_params

log = logging.getLogger(__name__)
OUT = sd.OUT
SPLIT_YEAR = 2023
NEG_SAMPLE = 15000
SILL = 0.10
# 지형: 공개DEM 4도엽(37608·37612·37705·37709) 병합으로 서울 반지하 후보 전부 상대고도·경사 있음
USE_TERRAIN = True
_TO5186 = pyproj.Transformer.from_crs(4326, 5186, always_xy=True)
# 강우 관측망: aws = 기상청 AWS만 / aws+seoul = 서울시 강우량계 48곳(OA-22715, src/calibration/seoul_gauges.py) 추가
RAIN_NET = os.environ.get("FLOOD_RAIN_NET", "aws")
GRID = {"i_design_mmh": [50, 75, 95], "pond_frac": [0.25, 0.5], "up_ratio": [1.0, 3.0]}


# ---------------------------------------------------------------- 준비
class Data:
    def __init__(self):
        from scipy.spatial import cKDTree
        self.b = pd.read_parquet(OUT / "buildings.parquet", columns=["bid", "approved", "x", "y", "rel_elev_m",
                                                                       "slope_pct"]).sort_values("bid")
        self.built = pd.to_datetime(self.b.approved, errors="coerce").to_numpy()
        self.labels = pd.read_parquet(OUT / "labels.parquet")
        self.events = sd.load_events()
        rng = np.random.default_rng(0)
        self.sample = np.sort(rng.choice(len(self.b), size=min(NEG_SAMPLE, len(self.b)), replace=False))
        st = rain_stations()
        sx, sy = _TO5186.transform(st.lon.to_numpy(), st.lat.to_numpy())
        self.stn = st.stn.astype(int).to_numpy()
        self.stn_xy = np.c_[sx, sy]
        ss = pd.read_csv(OUT / "sewer_stations.csv").dropna(subset=["lng"])
        qx, qy = _TO5186.transform(ss.lng.to_numpy(), ss.lat.to_numpy())
        self.sewer_ids = ss.station.tolist()
        dist, j = cKDTree(np.c_[qx, qy]).query(np.c_[self.b.x, self.b.y])
        self.sewer_idx = np.where(dist <= 1000, j, -1)
        self.h_crit = self._h_crit()
        log.info("건물 %d동, 사건별 무작위 표본 %d동, 수위계 %d곳(1km 안 연결 %d동)", len(self.b), len(self.sample),
                 len(self.sewer_ids), int((self.sewer_idx >= 0).sum()))

    def _h_crit(self):
        best = {}
        for _, e in self.events.iterrows():
            if int(e.event[:4]) >= SPLIT_YEAR or not e.sewer:
                continue
            for d in e.days.split():
                p = sd.SEWER / f"{d}.csv"
                if p.exists():
                    s = pd.read_csv(p)
                    for k, v in s.groupby("station").level_m.max().items():
                        best[k] = max(best.get(k, 0), v)
        known = [v for v in best.values() if v > 0]
        med = float(np.median(known)) if known else 1.0
        return np.array([best.get(s, 0) or med for s in self.sewer_ids])


class Prepared:
    """fit.evaluate가 쓰는 속성을 서울 표본으로 채운다.
    표본 = 그 사건 침수 건물 전부(가중치 1) + 서울 전체 무작위 표본 중 비침수(가중치 = 비침수 전체 ÷ 표본 비침수)."""

    def __init__(self, D: Data, e):
        self.ev = type("E", (), {"eid": e.event, "days": e.days.split()})()
        day0 = np.datetime64(pd.Timestamp(e.days.split()[0]))
        built = np.isnat(D.built) | (D.built <= day0)
        lab = D.labels[D.labels.event == e.event]
        pos = np.intersect1d(lab[lab.flood == 1].bid.to_numpy(), np.flatnonzero(built))
        exc = lab[lab.flood == -1].bid.to_numpy()
        neg_pool = built.copy()
        neg_pool[pos] = False
        neg_pool[exc] = False
        neg = D.sample[neg_pool[D.sample]]
        self.rows = np.concatenate([pos, neg])
        self.weight = np.concatenate([np.ones(len(pos)), np.full(len(neg), neg_pool.sum() / max(len(neg), 1))])
        self.bids = self.rows
        # 강우
        if not all((sd.RAIN10 / f"{d}.csv").exists() for d in self.ev.days):
            raise ValueError("10분 강우 아직 없음")
        rows = [r for d in self.ev.days for r in _rain(d)]
        if not rows:
            raise ValueError("강우 자료 없음")
        times = sorted({t for _, t, _ in rows})
        ti = {t: i for i, t in enumerate(times)}
        si = {s: i for i, s in enumerate(D.stn)}
        a = np.full((len(D.stn), len(times)), np.nan)
        for s, t, v in rows:
            if s in si:
                a[si[s], ti[t]] = v
        have = ~np.isnan(a).all(axis=1)
        a, sxy = a[have], D.stn_xy[have]
        wet = np.nan_to_num(window_sum(a, 18)).max(axis=0) > 0
        wet[0] = True
        keep = np.flatnonzero(wet)
        self.times = [times[i] for i in keep]
        bx, by = D.b.x.to_numpy()[self.rows], D.b.y.to_numpy()[self.rows]
        W = 1 / np.maximum(np.hypot(bx[:, None] - sxy[None, :, 0], by[:, None] - sxy[None, :, 1]), 50) ** 2

        def idw(x):
            ok = ~np.isnan(x)
            den = W @ ok.astype(float)
            return np.where(den > 0, (W @ np.nan_to_num(x)) / np.maximum(den, 1e-12), np.nan).astype(np.float16)
        self.P60 = idw(window_sum(a, 6)[:, keep])
        self.P10 = idw(window_sum(a, 1)[:, keep])
        self.I30 = None                         # SWMM 방식에서는 쓰지 않음
        tm = np.array([t.timestamp() for t in self.times])
        self.dt_min = np.concatenate([[10.0], np.diff(tm) / 60])
        # 하수관
        n, T = len(self.rows), len(self.times)
        st = D.sewer_idx[self.rows]
        obs = _sewer(self.ev.days) if e.sewer else None
        self.sewer_available = obs is not None and (st >= 0).any()
        self.util, self.rise, self.sewer_ex = np.float32(np.nan), np.float32(np.nan), np.float32(0)
        self.hc = np.where(st >= 0, D.h_crit[np.maximum(st, 0)], np.nan)
        if self.sewer_available:
            self.util = np.full((n, T), np.nan, dtype=np.float16)
            self.rise = np.full((n, T), np.nan, dtype=np.float16)
            for j in np.unique(st[st >= 0]):
                sid = D.sewer_ids[j]
                if sid in obs:
                    m = st == j
                    h0, r30 = sewer_at(obs[sid], self.times)
                    self.util[m] = h0 / D.h_crit[j]
                    self.rise[m] = r30
        self.rel = D.b.rel_elev_m.to_numpy()[self.rows] if USE_TERRAIN else np.full(n, np.nan)
        self.slope = D.b.slope_pct.to_numpy()[self.rows] if USE_TERRAIN else np.full(n, np.nan)
        self.sill = np.full(n, SILL)


def rain_stations() -> pd.DataFrame:
    st = sd.stations()[["stn", "lon", "lat"]]
    if RAIN_NET == "aws+seoul":
        from src.calibration import seoul_gauges
        st = pd.concat([st, seoul_gauges.stations()[["stn", "lon", "lat"]]], ignore_index=True)
    return st


def _rain(day):
    p = sd.RAIN10 / f"{day}.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    if RAIN_NET == "aws+seoul":
        from src.calibration import seoul_gauges
        g = seoul_gauges.OUT / f"{day}.csv"
        if g.exists():
            d = pd.concat([d, pd.read_csv(g)], ignore_index=True)
    return list(zip(d.stn.astype(int), pd.to_datetime(d.ts).dt.to_pydatetime(), d.rn10.astype(float)))


def _sewer(days):
    obs = {}
    for d in days:
        p = sd.SEWER / f"{d}.csv"
        if not p.exists():
            return None
        s = pd.read_csv(p)
        for sid, g in s.groupby("station"):
            obs.setdefault(sid, []).extend(zip(pd.to_datetime(g.ts).dt.to_pydatetime(), g.level_m.astype(float)))
    return obs


# ---------------------------------------------------------------- 계산·평가
def table(D: Data, preps, rm) -> pd.DataFrame:
    parts = []
    for p in preps:
        d = pd.DataFrame(evaluate(p, rm))
        d["weight"] = p.weight
        d["event"], d["year"] = p.ev.eid, int(p.ev.days[0][:4])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    lab = D.labels[D.labels.flood == 1][["bid", "event", "depth", "date", "time"]].assign(flood=1)
    d = d.merge(lab, on=["bid", "event"], how="left")
    d["flood"] = d["flood"].fillna(0).astype(int)
    d["train"] = d.year < SPLIT_YEAR
    return d


def wmetrics(d: pd.DataFrame, high=2) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, w, pred = d.flood.to_numpy(), d.weight.to_numpy(), (d.max_level >= high).to_numpy()
    tp, fp = float(w[pred & (y == 1)].sum()), float(w[pred & (y == 0)].sum())
    fn = float(w[~pred & (y == 1)].sum())
    out = {"rows": len(d), "positives": int(y.sum()), "events": int(d.event.nunique()),
           "recall": tp / (tp + fn) if tp + fn else None, "precision": tp / (tp + fp) if tp + fp else None}
    if y.sum() and (1 - y).sum():
        out["pr_auc"] = float(average_precision_score(y, d.max_score, sample_weight=w))
        out["roc_auc"] = float(roc_auc_score(y, d.max_score, sample_weight=w))
    f = d[d.flood == 1]
    if len(f) and f.depth.notna().any():
        out["depth_mae"] = float(np.nanmean(np.abs(f.max_depth - f.depth)))
    dry = d[~d.event.isin(d[d.flood == 1].event.unique())]
    out["false_alarm_dry_events"] = float(np.average(dry.max_level >= high, weights=dry.weight)) if len(dry) else None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def fit_weights(d: pd.DataFrame, prior: dict):
    """§35 로지스틱 회귀. 하수관 자료가 있는 보정 기간 사건만. 자료가 없는 요소(지형 등)는 사전 가중치 비율 유지."""
    from sklearn.linear_model import LogisticRegression
    tr = d[d.train & d.s_sewer.notna()]
    use = [k for k in COMPS if tr[f"s_{k}"].notna().mean() > 0.5 and tr[f"s_{k}"].std() > 0]
    X = tr[[f"s_{k}" for k in use]].fillna(tr[[f"s_{k}" for k in use]].mean())
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X.to_numpy() / 100, tr.flood,
                                                                              sample_weight=tr.weight)
    coef = dict(zip(use, lr.coef_[0]))
    fixed = {k: prior[k] for k in COMPS if k not in use}
    rest = 1 - sum(fixed.values())
    pos = {k: max(0.0, v) for k, v in coef.items()}
    tot = sum(pos.values()) or 1
    return ({**fixed, **{k: round(rest * v / tot, 3) for k, v in pos.items()}},
            {"rows": len(tr), "positives": int(tr.flood.sum()), "events": int(tr.event.nunique()), "used": use,
             "coef": {k: round(float(v), 3) for k, v in coef.items()}, "intercept": round(float(lr.intercept_[0]), 3)})


def pick_levels(d: pd.DataFrame):
    from sklearn.metrics import precision_recall_curve
    tr = d[d.train]
    pre, rec, thr = precision_recall_curve(tr.flood, tr.max_score, sample_weight=tr.weight)
    pre, rec = pre[:-1], rec[:-1]
    f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-9)
    t_mod = float(thr[np.flatnonzero(rec >= 0.9).max()])
    t_high = float(thr[np.argmax(f1)])
    ok = np.flatnonzero(rec >= 0.3)
    t_vh = float(thr[ok[np.argmax(pre[ok])]])
    lv = [round(t_mod, 1), round(max(t_high, t_mod + 5), 1), round(max(t_vh, t_high + 5, t_mod + 10), 1)]
    return lv, {"moderate_recall90": round(t_mod, 1), "high_bestF1": round(t_high, 1), "f1": round(float(f1.max()), 3),
                "very_high": round(t_vh, 1), "precision_at_vh": round(float(pre[ok].max()), 3)}


def lead_times(d: pd.DataFrame) -> dict:
    """§41 침수흔적에 피해 시각이 있는 건물(2025년): 처음 '경고' 이상이 된 시각 − 피해 시각."""
    f = d[(d.flood == 1) & d.time.notna() & (d.first_high != "")].copy()
    if f.empty:
        return {}
    obs = pd.to_datetime(f.date.astype(str) + f.time.astype(str).str.zfill(4), format="%Y%m%d%H%M", errors="coerce")
    lead = (obs - pd.to_datetime(f.first_high)).dt.total_seconds() / 60
    lead = lead.dropna()
    n_all = int(((d.flood == 1) & d.time.notna()).sum())
    return {"buildings_with_time": n_all, "warned": len(lead), "lead_min_median": float(lead.median()),
            "lead_min_p25": float(lead.quantile(.25)), "lead_min_p75": float(lead.quantile(.75)),
            "warned_before_damage_share": float((lead > 0).mean())}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    D = Data()
    rm0 = copy.deepcopy(load_params()["risk_model"])
    preps, skipped = [], []
    for _, e in D.events.iterrows():
        try:
            preps.append(Prepared(D, e))
        except ValueError as err:
            skipped.append(e.event)
            log.warning("%s 건너뜀: %s", e.event, err)
    log.info("사건 %d개 준비 (건너뜀 %d)", len(preps), len(skipped))
    rep = {"date": date.today().isoformat(), "split": f"보정 2010~{SPLIT_YEAR - 1} / 시험 {SPLIT_YEAR}~2025",
           "sample": {"buildings": len(D.b), "random_sample_per_event": len(D.sample)}, "skipped_events": skipped}

    d0 = table(D, preps, rm0)
    rep["baseline_v0.1_swmm"] = {"calibration": wmetrics(d0[d0.train]), "test": wmetrics(d0[~d0.train])}
    log.info("기준선 %s", rep["baseline_v0.1_swmm"])
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    rows = []
    tr = [p for p in preps if int(p.ev.days[0][:4]) < SPLIT_YEAR]
    for vals in itertools.product(*GRID.values()):
        r = copy.deepcopy(rm0)
        r["surface"]["swmm"].update(dict(zip(GRID, vals)))
        m = wmetrics(table(D, tr, r))
        rows.append({**dict(zip(GRID, vals)), **m})
        log.info("격자 %s → PR-AUC %s, 헛경보 %s", dict(zip(GRID, vals)), m.get("pr_auc"), m.get("false_alarm_dry_events"))
    g = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    g.to_csv(OUT / "grid_swmm.csv", index=False, encoding="utf-8-sig")
    rm = copy.deepcopy(rm0)
    rm["surface"]["swmm"].update({k: float(g.iloc[0][k]) for k in GRID})

    d1 = table(D, preps, rm)
    w, wi = fit_weights(d1, rm0["weights"])
    rm["weights"] = w
    d2 = table(D, preps, rm)
    lv, li = pick_levels(d2)
    rm["levels"] = lv
    d3 = table(D, preps, rm)
    rep["calibrated"] = {"swmm": {k: rm["surface"]["swmm"][k] for k in GRID}, "weights": w, "weights_fit": wi,
                         "levels": lv, "levels_fit": li,
                         "calibration": wmetrics(d3[d3.train]), "test": wmetrics(d3[~d3.train])}
    rep["lead_time_2025"] = {"v0.1": lead_times(d0[~d0.train]), "calibrated": lead_times(d3[~d3.train])}
    ev = d3.groupby("event").apply(lambda x: pd.Series({
        "positives": int(x.flood.sum()), "high_share_w": round(float(np.average(x.max_level >= 2, weights=x.weight)), 4),
        "recall": round(float((x[x.flood == 1].max_level >= 2).mean()), 3) if x.flood.sum() else None}))
    ev0 = d0.groupby("event").apply(lambda x: round(float(np.average(x.max_level >= 2, weights=x.weight)), 4))
    ev["high_share_w_v01"] = ev0
    ev.to_csv(OUT / "events_result.csv", encoding="utf-8-sig")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("보정 후 %s", rep["calibrated"])
    log.info("선행시간 %s", rep["lead_time_2025"])


if __name__ == "__main__":
    main()
