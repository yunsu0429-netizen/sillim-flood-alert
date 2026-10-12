"""칼리브레이션 실행 (prototype.md §42 STEP 4~10).

simulate.py를 빠른 형태로 나눔: 사건마다 관측소 단위 입력(강우 3곳·수위계)을 한 번 준비(prepare)하고,
계수만 바꿔 반지하 후보 건물 × 10분 시각을 다시 계산(evaluate)한다. 수식은 risk_model.py 함수를 그대로 쓴다.

STEP 4~5  v0.1 그대로 예측 → 정답과 비교 (기준선)
STEP 6    α·β·γ·배수능력 격자 탐색 (보정 기간 PR-AUC 최대, 침수심 MAE 참고)
STEP 7    요소 점수 → 로지스틱 회귀(§35) → 양수 계수를 정규화해 가중치
STEP 8    등급 경계: 주의 = 재현율 0.95, 경고 = F1 최대, 위험 = 정밀도 최대(재현율 0.3 이상) 점수
          + 개구부 초과 override 검증(§32.1), 점수 → 침수확률(Platt, §36)
STEP 9~10 2021~2025 사건으로 독립 시험

누설 방지: H_critical(만관 수위)을 2022-08-08 최고수위로 두면 시험 사건 정보가 들어가므로,
보정 중에는 보정 기간(2020년) 수위 최댓값을 기준으로 쓴다.

실행: .venv\\Scripts\\python -m src.calibration.fit
"""
import copy
import itertools
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.calibration.calibrate import OUT, SPLIT_YEAR, Context, events, metrics
from src.calibration.simulate import load_sewer, sewer_at, station_rain, window_sum
from src.config import ROOT
from src.engine import swmm_surface
from src.engine.risk_model import (classify, excess_over, opening_score, rain_score, sewer_score, surface_depth,
                                   terrain_score, weighted_score)

log = logging.getLogger(__name__)
COMPS = ["opening", "sewer", "rain", "terrain"]


# ---------------------------------------------------------------- 준비
def train_h_crit(ctx: Context, evs) -> tuple[np.ndarray, list]:
    """보정 기간 사건의 수위계별 최고수위. 자료 없는 수위계는 그 값들의 중앙값."""
    best = {}
    for ev in evs:
        if int(ev.days[0][:4]) >= SPLIT_YEAR:
            continue
        obs = load_sewer(ev, ctx.stations)
        for s, rows in (obs or {}).items():
            best[s] = max(best.get(s, 0.0), max(v for _, v in rows))
    med = float(np.median([v for v in best.values() if v > 0])) if best else 1.0
    h = np.array([best.get(s["station"], 0.0) or med for s in ctx.stations])
    src = ["보정기간 최고수위" if best.get(s["station"], 0) > 0 else "중앙값 대체" for s in ctx.stations]
    return h, src


class Prepared:
    """한 사건의 반지하 후보 건물 입력 (계수와 무관한 부분)."""

    def __init__(self, ctx: Context, ev, bids: np.ndarray, h_crit: np.ndarray):
        self.ev, self.bids = ev, bids
        times, sr = station_rain(ev)
        # 3곳 모두 최근 3시간 비가 없는 시각은 결과(사건 최댓값)에 영향이 거의 없어 계산에서 뺀다 (속도)
        wet = np.nan_to_num(window_sum(sr, 18)).max(axis=0) > 0
        wet[0] = True
        keep = np.flatnonzero(wet)
        w = ctx.rain_w[bids]
        p30, p60 = window_sum(sr, 3)[:, keep], window_sum(sr, 6)[:, keep]
        self.times = [times[i] for i in keep]

        def idw(x):
            ok = ~np.isnan(x)
            den = w @ ok.astype(float)
            return np.where(den > 0, (w @ np.nan_to_num(x)) / np.maximum(den, 1e-9), np.nan)
        self.P60 = idw(p60).astype(np.float32)
        self.P10 = idw(window_sum(sr, 1)[:, keep]).astype(np.float32)          # SWMM 적분용 10분 강수
        tm = np.array([t.timestamp() for t in self.times])
        self.dt_min = np.concatenate([[10.0], np.diff(tm) / 60])
        self.I30 = np.nan_to_num(idw(p30) * 2).astype(np.float32)
        n, T = len(bids), len(self.times)
        st = ctx.sewer_link(ev)[bids]
        obs = load_sewer(ev, ctx.stations)
        self.sewer_available = obs is not None and (st >= 0).any()
        # 하수관 자료가 없는 사건은 배열 대신 스칼라 NaN/0 (메모리 절약, 계산은 브로드캐스트)
        self.util, self.rise, self.sewer_ex = np.float32(np.nan), np.float32(np.nan), np.float32(0)
        if self.sewer_available:
            self.util = np.full((n, T), np.nan, dtype=np.float32)
            self.rise = np.full((n, T), np.nan, dtype=np.float32)
            for j, s in enumerate(ctx.stations):
                m = st == j
                if m.any():
                    h0, r30 = sewer_at(obs.get(s["station"], []), self.times)
                    self.util[m] = h0 / h_crit[j]
                    self.rise[m] = r30
            hc = h_crit[np.maximum(st, 0)][:, None].astype(np.float32)
            self.sewer_ex = np.where(np.isnan(self.util), 0, np.maximum(0, np.nan_to_num(self.util) - 1) * hc
                                     ).astype(np.float32)
        self.hc = np.where(st >= 0, h_crit[np.maximum(st, 0)], np.nan)
        self.slope = ctx.static["slope_pct"].to_numpy(dtype=float)[bids]
        rel = ctx.static["rel_elev_m"].to_numpy()[bids]
        self.rel = rel
        self.sill = ctx.engine.model.sill[bids]


def prepare_all(ctx: Context, evs, h_crit):
    built = pd.to_datetime(ctx.static["approved"], errors="coerce")
    out = []
    for ev in evs:
        day = pd.Timestamp(ev.days[0])
        bids = np.flatnonzero(ctx.banjiha & (built.isna() | (built <= day)).to_numpy())
        out.append(Prepared(ctx, ev, bids, h_crit))
        log.info("준비 %s  건물 %d  시각 %d  하수관 %s", ev.eid, len(bids), len(out[-1].times),
                 "있음" if out[-1].sewer_available else "없음")
    return out


# ---------------------------------------------------------------- 계산
def evaluate(p: Prepared, rm: dict) -> dict:
    sf = rm["surface"]
    lowland = np.nan_to_num(np.clip(-p.rel / sf["lowland_full_m"], 0, 1))
    if sf.get("method") == "swmm":
        depth = swmm_surface.simulate(sf, p.P10, p.dt_min, p.util, p.hc, p.slope, lowland)
    else:
        depth = surface_depth(p.sewer_ex, p.I30, lowland[:, None], sf)
    ex = excess_over(depth, p.sill[:, None])
    T = len(p.times)
    key = json.dumps(rm["scores"], sort_keys=True)
    if getattr(p, "_ckey", None) != key:          # 계수 탐색 중엔 하수관·강우·지형 점수가 그대로라 한 번만 계산
        p._ckey = key
        p._fixed = {"sewer": sewer_score(p.util, rm["scores"]["sewer"]).astype(np.float32),
                    "rain": rain_score(p.P60, rm["scores"]["rain60"]).astype(np.float32),
                    "terrain": terrain_score(p.rel, rm["scores"]["terrain"])[:, None]}
    comps = {"opening": opening_score(ex), **p._fixed}
    score = weighted_score(comps, rm["weights"])
    level = classify(score, ex, rm["levels"], rm["override"], np.broadcast_to(p.rise, score.shape), p.P60)
    amax = np.nanargmax(np.where(np.isnan(score), -1, score), axis=1)
    r = np.arange(len(p.bids))
    first = np.where((level >= 2).any(axis=1), np.argmax(level >= 2, axis=1), -1)
    return {"bid": p.bids, "max_score": score[r, amax], "max_level": level.max(axis=1),
            "max_depth": depth.max(axis=1), "max_excess": ex.max(axis=1),
            # 점수가 가장 높았던 시각의 요소 점수 (로지스틱 회귀 입력)
            **{f"s_{k}": np.broadcast_to(comps[k], score.shape)[r, amax] for k in COMPS},
            "max_P60": np.nanmax(np.nan_to_num(p.P60), axis=1),
            "max_util": (np.where(np.isnan(p.util).all(axis=1), np.nan, np.nanmax(np.nan_to_num(p.util, nan=-1), axis=1))
                         if p.sewer_available else np.full(len(p.bids), np.nan)),
            "first_high": np.array([p.times[i].isoformat(timespec="minutes") if i >= 0 else "" for i in first])}


def table(ctx: Context, preps, rm) -> pd.DataFrame:
    parts = []
    for p in preps:
        d = pd.DataFrame(evaluate(p, rm))
        d["event"], d["year"] = p.ev.eid, int(p.ev.days[0][:4])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    lab = ctx.labels[["bid", "event", "flood", "depth", "label_kind"]]
    d = d.merge(lab, on=["bid", "event"], how="left")
    d["flood"] = d["flood"].fillna(0).astype(int)
    d["train"] = d["year"] < SPLIT_YEAR
    return d


def summary(d: pd.DataFrame, levels_high=2) -> dict:
    out = metrics(d.flood, d.max_level >= levels_high, d.max_score)
    f = d[d.flood == 1]
    if len(f):
        out["depth_mae"] = round(float(np.mean(np.abs(f.max_depth - f.depth))), 3)
        out["depth_bias"] = round(float(np.mean(f.max_depth - f.depth)), 3)
    # 침수 없는 호우에서 '경고' 이상이 된 반지하 비율 (헛경보)
    dry = d[~d.event.isin(d[d.flood == 1].event.unique())]
    out["false_alarm_dry_events"] = round(float((dry.max_level >= levels_high).mean()), 4) if len(dry) else None
    return out


# ---------------------------------------------------------------- STEP 6~8
def grid_surface(ctx, preps, rm):
    tr = [p for p in preps if int(p.ev.days[0][:4]) < SPLIT_YEAR]
    rows = []
    for a, b, g, drain in itertools.product([0.25, 0.5, 1.0], [0.002, 0.005, 0.01, 0.02],
                                            [0.0, 0.05, 0.15, 0.3], [30, 50, 70, 90]):
        r = copy.deepcopy(rm)
        r["surface"].update(alpha=a, beta=b, gamma=g, rain_drain_capacity_mmh=drain)
        d = table(ctx, tr, r)
        m = summary(d)
        rows.append({"alpha": a, "beta": b, "gamma": g, "drain": drain, **m})
    g = pd.DataFrame(rows).sort_values(["pr_auc", "depth_mae"], ascending=[False, True])
    return g


SWMM_GRID = {"i_design_mmh": [50, 75, 95], "pond_frac": [0.25, 0.5, 1.0], "up_ratio": [0.0, 1.0, 3.0],
             "orifice_cda_m2": [0.005, 0.02]}


def grid_swmm(ctx, preps, rm):
    """SWMM 계수 격자 탐색. PR-AUC(소수 셋째 자리) 최대, 같으면 침수심 MAE 최소 (§31.1 '실제 침수수심에 맞도록')."""
    tr = [p for p in preps if int(p.ev.days[0][:4]) < SPLIT_YEAR]
    rows = []
    keys = list(SWMM_GRID)
    for vals in itertools.product(*SWMM_GRID.values()):
        r = copy.deepcopy(rm)
        r["surface"]["swmm"].update(dict(zip(keys, vals)))
        m = summary(table(ctx, tr, r))
        rows.append({**dict(zip(keys, vals)), **m})
        log.info("SWMM %s → PR-AUC %s MAE %s", dict(zip(keys, vals)), m.get("pr_auc"), m.get("depth_mae"))
    g = pd.DataFrame(rows)
    g["pr_auc_r"] = g.pr_auc.round(3)
    return g.sort_values(["pr_auc_r", "depth_mae"], ascending=[False, True]).drop(columns="pr_auc_r")


def fit_weights(d: pd.DataFrame, prior: dict) -> tuple[dict, dict]:
    """§35 로지스틱 회귀. 보정 기간에 자료가 없는 요소(하수관 등)는 사전 가중치를 유지하고 나머지만 비율로 나눈다."""
    from sklearn.linear_model import LogisticRegression
    tr = d[d.train]
    usable = [k for k in COMPS if tr[f"s_{k}"].notna().mean() > 0.5 and tr[f"s_{k}"].std() > 0]
    X = tr[[f"s_{k}" for k in usable]].fillna(0).to_numpy() / 100
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X, tr.flood)
    coef = dict(zip(usable, lr.coef_[0]))
    pos = {k: max(0.0, v) for k, v in coef.items()}
    fixed = {k: prior[k] for k in COMPS if k not in usable}
    rest = 1 - sum(fixed.values())
    tot = sum(pos.values()) or 1
    w = {**fixed, **{k: round(rest * v / tot, 3) for k, v in pos.items()}}
    return w, {"usable": usable, "coef": {k: round(float(v), 3) for k, v in coef.items()},
               "intercept": round(float(lr.intercept_[0]), 3), "fixed_prior": fixed}


def pick_levels(d: pd.DataFrame) -> tuple[list, dict]:
    """§31.3 보정 기간 점수 분포에서 등급 경계를 고른다."""
    from sklearn.metrics import precision_recall_curve
    tr = d[d.train]
    pre, rec, thr = precision_recall_curve(tr.flood, tr.max_score)
    pre, rec = pre[:-1], rec[:-1]
    f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-9)
    t_mod = float(thr[np.flatnonzero(rec >= 0.95).max()])
    t_high = float(thr[np.argmax(f1)])
    ok = np.flatnonzero(rec >= 0.3)
    t_vh = float(thr[ok[np.argmax(pre[ok])]])
    lv = sorted({round(t_mod, 1), round(max(t_high, t_mod + 1), 1), round(max(t_vh, t_high + 1), 1)})
    info = {"moderate": {"score": round(t_mod, 1), "recall_target": 0.95},
            "high": {"score": round(t_high, 1), "f1": round(float(f1.max()), 3)},
            "very_high": {"score": round(t_vh, 1), "precision": round(float(pre[ok].max()), 3)}}
    return lv, info


def opening_curve(d: pd.DataFrame) -> list:
    """§32.1 개구부 초과수심 구간별 침수 발생률 (보정 기간)."""
    tr = d[d.train]
    bins = [-0.001, 0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 9]
    g = tr.groupby(pd.cut(tr.max_excess, bins), observed=True).flood.agg(["size", "mean"])
    return [{"excess_m": str(k), "n": int(r["size"]), "flood_rate": round(float(r["mean"]), 4)} for k, r in g.iterrows()]


def rate_curve(d, col, bins) -> list:
    tr = d[d.train & d[col].notna()]
    g = tr.groupby(pd.cut(tr[col], bins), observed=True).flood.agg(["size", "mean"])
    return [{col: str(k), "n": int(r["size"]), "flood_rate": round(float(r["mean"]), 4)} for k, r in g.iterrows()]


def platt(d: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    tr = d[d.train]
    lr = LogisticRegression().fit(tr[["max_score"]].to_numpy() / 100, tr.flood)
    return {"a": round(float(lr.coef_[0][0]), 3), "b": round(float(lr.intercept_[0]), 3)}


def prob(score, pl):
    return 1 / (1 + np.exp(-(pl["a"] * np.asarray(score) / 100 + pl["b"])))


# ---------------------------------------------------------------- main
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    evs = events()
    rm0 = copy.deepcopy(ctx.p["risk_model"])
    h_crit, h_src = train_h_crit(ctx, evs)
    log.info("H_crit(보정기간 기준): %s", {s["station"]: round(h, 2) for s, h, src in zip(ctx.stations, h_crit, h_src)
                                         if src != "중앙값 대체"})
    preps = prepare_all(ctx, evs, h_crit)
    tag = rm0["surface"].get("method", "empirical")
    report = {"model_version": f"v0.2-{tag}", "surface_method": tag, "calibration_date": date.today().isoformat(),
              "split": {"calibration": f"2010~{SPLIT_YEAR - 1}", "test": f"{SPLIT_YEAR}~2025"},
              "events": {"calibration": [p.ev.eid for p in preps if int(p.ev.days[0][:4]) < SPLIT_YEAR],
                         "test": [p.ev.eid for p in preps if int(p.ev.days[0][:4]) >= SPLIT_YEAR],
                         "flood_events": [p.ev.eid for p in preps if p.ev.flood]},
              "h_critical_source": "보정기간(2020) 수위계별 최고수위, 없으면 중앙값"}

    # STEP 4~5 기준선
    d0 = table(ctx, preps, rm0)
    report["baseline_v0.1"] = {"calibration": summary(d0[d0.train]), "test": summary(d0[~d0.train])}
    log.info("기준선 %s", report["baseline_v0.1"])

    # STEP 6
    rm = copy.deepcopy(rm0)
    if rm0["surface"].get("method") == "swmm":
        g = grid_swmm(ctx, preps, rm0)
        best = g.iloc[0]
        rm["surface"]["swmm"].update({k: float(best[k]) for k in SWMM_GRID})
    else:
        g = grid_surface(ctx, preps, rm0)
        best = g.iloc[0]
        rm["surface"].update(alpha=float(best.alpha), beta=float(best.beta), gamma=float(best.gamma),
                             rain_drain_capacity_mmh=float(best.drain))
    g.to_csv(OUT / f"grid_surface_{rm0['surface'].get('method', 'empirical')}.csv", index=False, encoding="utf-8-sig")
    log.info("STEP6 최적 %s", best.to_dict())

    # STEP 7
    d1 = table(ctx, preps, rm)
    w, wi = fit_weights(d1, rm0["weights"])
    rm["weights"] = w
    log.info("STEP7 가중치 %s  %s", w, wi)

    # STEP 8
    d2 = table(ctx, preps, rm)
    lv, li = pick_levels(d2)
    rm["levels"] = lv
    d3 = table(ctx, preps, rm)
    pl = platt(d3)
    d3["prob"] = prob(d3.max_score, pl)
    curves = {"opening_excess": opening_curve(d3),
              "rain60_mm": rate_curve(d3, "max_P60", [0, 10, 20, 30, 40, 50, 60, 80, 200]),
              "sewer_util": rate_curve(d3, "max_util", [-1, 0.5, 0.7, 0.9, 1.0, 1.2, 9])}

    # STEP 9~10
    cal, test = d3[d3.train], d3[~d3.train]
    report["calibrated"] = {
        "surface": rm["surface"],
        "weights": w, "weights_fit": wi, "levels": lv, "levels_fit": li, "probability_platt": pl,
        "curves": curves,
        "calibration": {**summary(cal), "brier": round(float(np.mean((cal.prob - cal.flood) ** 2)), 4)},
        "test": {**summary(test), "brier": round(float(np.mean((test.prob - test.flood) ** 2)), 4)}}
    # 사건별 경고 이상 비율
    ev = d3.groupby("event").agg(flood_event=("flood", "max"), n=("flood", "size"), positives=("flood", "sum"),
                                 high_share=("max_level", lambda s: round(float((s >= 2).mean()), 4)),
                                 mean_score=("max_score", lambda s: round(float(s.mean()), 1)))
    ev0 = d0.groupby("event").max_level.apply(lambda s: round(float((s >= 2).mean()), 4)).rename("high_share_v01")
    ev = ev.join(ev0)
    ev.to_csv(OUT / f"events_summary_{tag}.csv", encoding="utf-8-sig")
    report["by_event"] = ev.reset_index().to_dict("records")
    # 419-1 선행시간 (§41: 보도 기준 21시대 침수)
    b419 = ctx.static.index[ctx.static.jibun == "419-1"]
    for name, dd in (("v0.1", d0), ("v0.2", d3)):
        r = dd[(dd.event == "2022-08-08") & dd.bid.isin(b419)]
        report.setdefault("case_419_1", {})[name] = r[["max_score", "max_level", "first_high"]].to_dict("records")
    d3.to_parquet(OUT / f"results_v02_{tag}.parquet")
    (OUT / f"calibration_report_{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                                 encoding="utf-8")
    rm = json.loads(json.dumps(rm, default=float))
    yaml.safe_dump({"risk_model": rm, "h_critical_calibration": dict(zip([s["station"] for s in ctx.stations],
                                                                        map(float, h_crit)))},
                   open(OUT / f"risk_model_v02_{tag}.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    log.info("완료: %s", json.dumps({k: report[k] for k in ("baseline_v0.1",)}, ensure_ascii=False))
    log.info("보정 후: cal %s / test %s", report["calibrated"]["calibration"], report["calibrated"]["test"])


if __name__ == "__main__":
    main()
