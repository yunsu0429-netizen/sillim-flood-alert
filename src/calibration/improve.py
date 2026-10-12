"""선행연구 기반 경보 개선 시험 (서울 전역 표본, 보정 2010~2022 / 시험 2023~2025).

단계 (docs/references.md에 수식·문헌 정리)
  base  서울 보정 모델(SWMM 50mm/h·0.5·3배, 회귀 가중치)로 건물×사건 표 생성.
        행마다 최대 점수·요소 점수·지속시간별 최대 강우강도, 그리고 점수 경계 θ마다 처음 넘은 시각을 저장해
        경계를 바꿔도 다시 모의하지 않고 경보 시각을 구할 수 있게 한다.
  A     등급 경계를 비용-손실 기대가치(Murphy 1977; Thiboult 외 2017)로 재선정 (보정 기간 POD ≥ 0.5 조건)
  B     지속시간별 강우 임계값(Jang 2015; Tian 외 2019)을 ROC·Youden 지수로 정해 강우 점수 교체 → 표 다시 생성
  D     건물 정적 취약성 p_static (HistGradientBoosting, 100m 블록 공간 교차검증; Zischg 외 2018, Qin 외 2025)
        → 로지스틱 결합 logit P = g0 + g1·R/100 + g2·logit p_static
  F     확률 보정(Platt 1999) + Brier 점수(Brier 1950)

실행: .venv\\Scripts\\python -m src.calibration.improve base A B D F
"""
import argparse
import copy
import json
import logging
import os

import numpy as np
import pandas as pd

from src.calibration import fit_seoul as fs
from src.calibration import seoul_data as sd
from src.calibration.simulate import window_sum
from src.config import load_params
from src.engine import swmm_surface
from src.engine.risk_model import excess_over, opening_score, rain_score, sewer_score, terrain_score, weighted_score

log = logging.getLogger(__name__)
OUT = sd.OUT / "improve"
THETAS = list(range(10, 100, 5))
DURS = {10: 1, 30: 3, 60: 6, 180: 18}            # 분 → 10분 칸 수
CL_RATIOS = [0.05, 0.1, 0.2, 0.3]
EPOCH = pd.Timestamp("1970-01-01")                 # 비용/손실 비 (대피·물막이판 비용 ÷ 반지하 침수 피해)


# ---------------------------------------------------------------- 모델 설정
def seoul_model() -> dict:
    """서울 보정 결과(report.json)의 SWMM 계수·가중치를 v0.1 설정 위에 얹는다."""
    rm = copy.deepcopy(load_params()["risk_model"])
    rep = json.loads((sd.OUT / "report.json").read_text(encoding="utf-8"))
    rm["surface"]["swmm"].update(rep["calibrated"]["swmm"])
    rm["weights"] = {k: float(v) for k, v in rep["calibrated"]["weights"].items()}
    return rm


def rain_ratio_score(I: dict, thr: dict) -> np.ndarray:
    """B: 지속시간별 임계 대비 비율의 최댓값 → 0~100점. 비율 0.5에서 0점, 1.0(임계)에서 50점, 1.5에서 100점."""
    ratio = np.max(np.stack([I[d] / thr[d] for d in thr]), axis=0)
    return np.clip((ratio - 0.5) / 1.0, 0, 1) * 100


# ---------------------------------------------------------------- 평가기
def evaluate2(p, rm, rain_thr=None) -> dict:
    sf = rm["surface"]
    lowland = np.nan_to_num(np.clip(-p.rel / sf["lowland_full_m"], 0, 1))
    depth = swmm_surface.simulate(sf, p.P10, p.dt_min, p.util, p.hc, p.slope, lowland).astype(np.float32)
    ex = excess_over(depth, p.sill[:, None])
    T = len(p.times)
    p10 = np.nan_to_num(p.P10.astype(np.float32))
    I = {d: window_sum(p10, k) * (60.0 / d) for d, k in DURS.items()}          # mm/h
    rain = rain_ratio_score(I, rain_thr) if rain_thr else rain_score(p.P60, rm["scores"]["rain60"])
    comps = {"opening": opening_score(ex), "sewer": sewer_score(p.util, rm["scores"]["sewer"]),
             "rain": rain, "terrain": terrain_score(p.rel, rm["scores"]["terrain"])[:, None]}
    score = weighted_score(comps, rm["weights"])
    score = np.where(np.isnan(score), -1, score)
    n = len(p.bids)
    r = np.arange(n)
    amax = score.argmax(axis=1)
    tmin = np.array([(pd.Timestamp(t) - EPOCH).total_seconds() / 60 for t in p.times])   # 분, 피해시각과 같은 기준

    def first(mask):
        any_ = mask.any(axis=1)
        return np.where(any_, tmin[mask.argmax(axis=1)], np.nan)
    ov = rm["override"]
    rise = np.broadcast_to(np.asarray(p.rise, dtype=np.float32), score.shape)
    crit = (np.nan_to_num(rise, nan=-1) >= ov["critical_rise_m_per_min"]) & \
           (np.nan_to_num(p.P60.astype(np.float32), nan=-1) >= ov["heavy_rain60_mm"])
    out = {"bid": p.bids, "max_score": score[r, amax], "max_depth": depth.max(axis=1), "max_excess": ex.max(axis=1),
           **{f"s_{k}": np.broadcast_to(comps[k], score.shape)[r, amax] for k in comps},
           **{f"I{d}": I[d].max(axis=1) for d in DURS},
           "t_excess": first(ex > 0), "t_excess30": first(ex >= ov["opening_very_high_m"]), "t_override": first(crit)}
    for th in THETAS:
        out[f"t_{th}"] = first(score >= th)
    return out


def build(D, preps, rm, rain_thr=None) -> pd.DataFrame:
    parts = []
    for p in preps:
        d = pd.DataFrame(evaluate2(p, rm, rain_thr))
        d["weight"] = p.weight
        d["event"], d["year"] = p.ev.eid, int(p.ev.days[0][:4])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    lab = D.labels[D.labels.flood == 1][["bid", "event", "depth", "date", "time"]].assign(flood=1)
    d = d.merge(lab, on=["bid", "event"], how="left")
    d["flood"] = d["flood"].fillna(0).astype(int)
    d["train"] = d.year < fs.SPLIT_YEAR
    obs = pd.to_datetime(d.date.astype(str) + d.time.astype(str).str.zfill(4), format="%Y%m%d%H%M", errors="coerce")
    d["t_damage"] = (obs - EPOCH).dt.total_seconds() / 60                               # 분 (epoch)
    return d


# ---------------------------------------------------------------- 경보 판정·지표
def warn_time(d: pd.DataFrame, th: int) -> pd.Series:
    """'경고' 이상 처음 시각 = min(점수 ≥ θ, 개구부 초과 > 0, 하수관 급상승+호우) (risk_model.classify와 같은 규칙)."""
    return d[[f"t_{th}", "t_excess", "t_override"]].min(axis=1)


def verify(d: pd.DataFrame, th: int) -> dict:
    """§39·§41 POD·FAR·CSI(건물 기준, 비침수는 표본 가중치), 피해 시각 있는 건물의 선행시간."""
    w, y = d.weight.to_numpy(), d.flood.to_numpy()
    warn = warn_time(d, th).notna().to_numpy()
    H = w[warn & (y == 1)].sum(); M = w[~warn & (y == 1)].sum(); F = w[warn & (y == 0)].sum()
    out = {"theta": th, "POD": H / (H + M) if H + M else np.nan, "FAR": F / (H + F) if H + F else np.nan,
           "CSI": H / (H + M + F) if H + M + F else np.nan, "warned_share": float(np.average(warn, weights=w))}
    f = d[(d.flood == 1) & d.t_damage.notna()]
    if len(f):
        lead = (f.t_damage - warn_time(f, th)).dropna()
        out.update({"timed": len(f), "warned_before_damage": float((lead > 0).sum() / len(f)),
                    "lead_median_min": float(lead.median()) if len(lead) else np.nan})
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def value(d: pd.DataFrame, th: int, cl: float) -> float:
    """비용-손실 상대가치 V (Murphy 1977, Thiboult 외 2017): 경보 비용 C, 놓친 침수 손실 L, C/L = cl.
    E = C·(H+F) + L·M,  V = (E_무경보 − E) / (E_무경보 − E_완벽).  L=1로 정규화."""
    w, y = d.weight.to_numpy(), d.flood.to_numpy()
    warn = warn_time(d, th).notna().to_numpy()
    H = w[warn & (y == 1)].sum(); M = w[~warn & (y == 1)].sum(); F = w[warn & (y == 0)].sum()
    P = w[y == 1].sum()
    e = cl * (H + F) + M
    e_none, e_perfect = P, cl * P
    return float((e_none - e) / (e_none - e_perfect)) if e_none > e_perfect else np.nan


def step_A(d: pd.DataFrame) -> dict:
    tr, te = d[d.train], d[~d.train]
    rows = []
    for th in THETAS:
        v = verify(tr, th)
        rows.append({**v, **{f"V_cl{cl}": round(value(tr, th, cl), 4) for cl in CL_RATIOS}})
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "A_threshold_curve_train.csv", index=False, encoding="utf-8-sig")
    choice = {}
    for cl in CL_RATIOS:
        ok = g[g.POD >= 0.5]
        if ok.empty:
            continue
        th = int(ok.sort_values(f"V_cl{cl}", ascending=False).iloc[0].theta)
        choice[cl] = {"theta_high": th, "train": verify(tr, th), "test": verify(te, th)}
    return {"curve": rows, "by_cost_loss": choice, "v01_theta40": {"train": verify(tr, 40), "test": verify(te, 40)},
            "theta75_near_seoul_cal": {"train": verify(tr, 75), "test": verify(te, 75)}}


# ---------------------------------------------------------------- B
def step_B(d: pd.DataFrame) -> dict:
    """지속시간별 최대 강우강도로 침수/비침수 ROC → Youden J 최대 임계 (보정 기간만)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    tr = d[d.train]
    thr, info = {}, {}
    for dur in DURS:
        x = tr[f"I{dur}"].to_numpy()
        fpr, tpr, cut = roc_curve(tr.flood, x, sample_weight=tr.weight)
        j = tpr - fpr
        k = int(np.argmax(j))
        thr[dur] = float(cut[k])
        info[dur] = {"threshold_mmh": round(float(cut[k]), 1), "youden_J": round(float(j[k]), 3),
                     "auc": round(float(roc_auc_score(tr.flood, x, sample_weight=tr.weight)), 3)}
    return {"thresholds": thr, "info": info}


# ---------------------------------------------------------------- D
def static_features(D) -> pd.DataFrame:
    b = pd.read_parquet(sd.OUT / "buildings.parquet",
                        columns=["bid", "x", "y", "approved", "structure", "bldg_area_m2", "floors_up", "floors_down",
                                 "rel_elev_m", "rel_elev_100m", "rel_elev_1km", "slope_pct"])
    b["year_built"] = pd.to_datetime(b.approved, errors="coerce").dt.year
    b["masonry"] = b.structure.astype(str).str.contains("조적|벽돌|블록").astype(int)
    b["block"] = (b.x // 100).astype(int).astype(str) + "_" + (b.y // 100).astype(int).astype(str)
    if EXTRA:
        from src.calibration import static_extra
        b = b.merge(pd.read_parquet(static_extra.OUT)[["bid", *static_extra.FEATS]], on="bid", how="left")
    return b


def history(D, b, year) -> pd.DataFrame:
    """그 해 이전 침수흔적(시간 누설 방지): 건물 자신의 과거 침수 횟수, 같은 100m 블록의 과거 침수 동수."""
    lab = D.labels[(D.labels.flood == 1) & (D.labels.event.str[:4].astype(int) < year)]
    own = lab.groupby("bid").size()
    blk = b.set_index("bid").block
    nb = lab.assign(block=blk.reindex(lab.bid).to_numpy()).groupby("block").size()
    return pd.DataFrame({"hist_own": own.reindex(b.bid).fillna(0).to_numpy(),
                         "hist_block": nb.reindex(b.block).fillna(0).to_numpy()}, index=b.bid)


# 서울시 공공데이터 추가 특성(침수예상도·빗물펌프장·불투수율, src/calibration/static_extra.py) 사용 여부
EXTRA = os.environ.get("FLOOD_EXTRA_FEATS", "0") == "1"
FEATS = ["rel_elev_m", "rel_elev_100m", "rel_elev_1km", "slope_pct", "year_built", "masonry", "bldg_area_m2",
         "floors_up", "floors_down"] + (["fmap_depth_m", "fmap_dist_m", "pump_dist_m", "gu_imperv_pct"] if EXTRA else [])         + ["hist_own", "hist_block"]


def step_D(D, d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    b = static_features(D)
    hist = {y: history(D, b, y) for y in sorted(d.year.unique())}
    X = b.set_index("bid").loc[d.bid, FEATS[:-2]].reset_index(drop=True)
    H = pd.concat([hist[y].loc[g.bid].reset_index(drop=True).set_axis(g.index) for y, g in d.groupby("year")])
    X[["hist_own", "hist_block"]] = H.sort_index().to_numpy()
    X["block"] = b.set_index("bid").loc[d.bid, "block"].to_numpy()
    tr = d.train.to_numpy()
    y = d.flood.to_numpy()
    sw = np.where(y == 1, 1.0, d.weight.to_numpy())
    pos_w = sw[~tr & (y == 0)].sum() / max(1, (y[tr] == 1).sum())
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    # 100m 블록 공간 교차검증 (보정 기간 안에서)
    gkf = GroupKFold(n_splits=5)
    oof = np.full(tr.sum(), np.nan)
    Xt, yt, wt, gt = X[tr].drop(columns="block"), y[tr], sw[tr], X.block[tr]
    bal = np.where(yt == 1, wt * (wt[yt == 0].sum() / max(1, wt[yt == 1].sum())), wt)
    for a, b_ in gkf.split(Xt, yt, gt):
        m = mk().fit(Xt.iloc[a], yt[a], sample_weight=bal[a])
        oof[b_] = m.predict_proba(Xt.iloc[b_])[:, 1]
    cv = {"pr_auc": round(float(average_precision_score(yt, oof, sample_weight=wt)), 4),
          "roc_auc": round(float(roc_auc_score(yt, oof, sample_weight=wt)), 4)}
    model = mk().fit(Xt, yt, sample_weight=bal)
    p_static = model.predict_proba(X.drop(columns="block"))[:, 1]
    p_static[tr] = oof                                   # 보정 기간 결합 학습엔 표본 밖 예측만 사용
    d = d.copy()
    d["p_static"] = np.clip(p_static, 1e-4, 1 - 1e-4)
    # 결합: logit P = g0 + g1·R/100 + g2·logit p_static  (보정 기간 학습)
    Z = np.c_[d.max_score.clip(0) / 100, np.log(d.p_static / (1 - d.p_static))]
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Z[tr], y[tr], sample_weight=sw[tr])
    d["P_comb_raw"] = lr.decision_function(Z)
    info = {"static_cv_block100m": cv, "combine_coef": {"g1_score": round(float(lr.coef_[0][0]), 3),
            "g2_logit_pstatic": round(float(lr.coef_[0][1]), 3), "g0": round(float(lr.intercept_[0]), 3)},
            "static_test": {"pr_auc": round(float(average_precision_score(y[~tr], d.p_static[~tr], sample_weight=sw[~tr])), 4)}}
    return d, info


# ---------------------------------------------------------------- F
def step_F(d: pd.DataFrame, col: str) -> dict:
    """Platt 보정: P = σ(a·z + b) (보정 기간 적합, 실제 비율 그대로 — class weight 없이) → 시험 Brier·신뢰도."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    tr, te = d[d.train], d[~d.train]
    z = lambda x: x[[col]].to_numpy()
    sw = lambda x: np.where(x.flood == 1, 1.0, x.weight)
    lr = LogisticRegression(max_iter=2000).fit(z(tr), tr.flood, sample_weight=sw(tr))
    pt = lr.predict_proba(z(te))[:, 1]
    bins = pd.cut(pt, [0, .01, .05, .1, .2, .5, 1], include_lowest=True)
    rel = pd.DataFrame({"p": pt, "y": te.flood, "w": sw(te)}).groupby(bins, observed=True).apply(
        lambda g: pd.Series({"pred": np.average(g.p, weights=g.w), "obs": np.average(g.y, weights=g.w), "n": len(g)}))
    return {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0]),
            "test": {"brier": round(float(brier_score_loss(te.flood, pt, sample_weight=sw(te))), 5),
                     "pr_auc": round(float(average_precision_score(te.flood, pt, sample_weight=sw(te))), 4),
                     "roc_auc": round(float(roc_auc_score(te.flood, pt, sample_weight=sw(te))), 4)},
            "reliability_test": rel.round(4).reset_index().astype(str).to_dict("records")}


def score_metrics(d: pd.DataFrame, col: str) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {}
    for name, x in (("train", d[d.train]), ("test", d[~d.train])):
        sw = np.where(x.flood == 1, 1.0, x.weight)
        out[name] = {"pr_auc": round(float(average_precision_score(x.flood, x[col], sample_weight=sw)), 4),
                     "roc_auc": round(float(roc_auc_score(x.flood, x[col], sample_weight=sw)), 4)}
    return out


# ---------------------------------------------------------------- main
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["base", "A", "B", "D", "F"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rep_path = OUT / "report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
    save = lambda: rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    need_sim = {"base", "B"} & set(a.steps)
    D = fs.Data() if need_sim or "D" in a.steps else None
    preps = [fs.Prepared(D, e) for _, e in D.events.iterrows()] if need_sim else None
    rm = seoul_model()
    if "base" in a.steps:
        d0 = build(D, preps, rm)
        d0.to_parquet(OUT / "table_base.parquet")
        rep["base"] = {"score": score_metrics(d0, "max_score")}
        log.info("base %s", rep["base"]); save()
    d0 = pd.read_parquet(OUT / "table_base.parquet")
    if "A" in a.steps:
        rep["A"] = step_A(d0)
        log.info("A %s", {k: v for k, v in rep["A"].items() if k != "curve"}); save()
    if "B" in a.steps:
        b = step_B(d0)
        rep["B"] = {"thresholds": b["info"]}
        d1 = build(D, preps, rm, rain_thr=b["thresholds"])
        # 강우 점수가 바뀌었으니 가중치를 같은 방법(§35 로지스틱 회귀)으로 다시 맞춘다
        w, wi = fs.fit_weights(d1.assign(s_sewer=d1.s_sewer), rm["weights"])
        rm_b = copy.deepcopy(rm); rm_b["weights"] = w
        d1 = build(D, preps, rm_b, rain_thr=b["thresholds"])
        d1.to_parquet(OUT / "table_B.parquet")
        rep["B"].update({"weights": w, "weights_fit": wi, "score": score_metrics(d1, "max_score"),
                         "A_on_B": step_A(d1)["by_cost_loss"]})
        (OUT / "rain_thresholds.json").write_text(json.dumps(b["thresholds"]), encoding="utf-8")
        log.info("B %s", {k: v for k, v in rep["B"].items() if k != "A_on_B"}); save()
    tb = OUT / "table_B.parquet"
    if "D" in a.steps:
        D = D or fs.Data()
        src = pd.read_parquet(tb) if tb.exists() else d0
        d2, info = step_D(D, src)
        d2.to_parquet(OUT / "table_D.parquet")
        rep["D"] = {**info, "combined": score_metrics(d2, "P_comb_raw"), "score_only": score_metrics(d2, "max_score")}
        log.info("D %s", rep["D"]); save()
    if "F" in a.steps:
        td = OUT / "table_D.parquet"
        d3 = pd.read_parquet(td) if td.exists() else d0
        col = "P_comb_raw" if "P_comb_raw" in d3 else "max_score"
        rep["F"] = {"input": col, **step_F(d3, col)}
        log.info("F %s", {k: v for k, v in rep["F"].items() if k != "reliability_test"}); save()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- 운영용 내보내기 (D 채택)
def export_static(z_high: float = -1.4, table: str = "table_D_on_base.parquet", coef: dict | None = None,
                  out_name: str = "static_prior.parquet"):
    """D 채택분을 대시보드에서 쓰도록 내보낸다.
    p_static: 보정 기간(2010~2022)으로 학습한 모형 + 2025년까지 전체 침수 이력 특성 → 서울 반지하 후보 전체.
    건물별 '경고' 경계: R ≥ θ_i = 100·(z* − g0 − g2·logit p_i)/g1 (z* = 보정 기간 경보 비율이 40점 경계와 같아지는 값).
    대시보드는 Δ_i = θ_i − 40 을 20/40/70 경계 모두에 더한다."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    D = fs.Data()
    d = pd.read_parquet(OUT / table)
    coef = coef or json.loads((OUT / "D_F_on_base.json").read_text(encoding="utf-8"))["D_on_base"]["combine_coef"]
    b = static_features(D)
    hist_tr = {y: history(D, b, y) for y in sorted(d[d.train].year.unique())}
    tr = d[d.train].reset_index(drop=True)
    X = b.set_index("bid").loc[tr.bid, FEATS[:-2]].reset_index(drop=True)
    H = pd.concat([hist_tr[y].loc[g.bid].reset_index(drop=True).set_axis(g.index) for y, g in tr.groupby("year")])
    X[["hist_own", "hist_block"]] = H.sort_index().to_numpy()
    y = tr.flood.to_numpy()
    sw = np.where(y == 1, 1.0, tr.weight.to_numpy())
    bal = np.where(y == 1, sw * (sw[y == 0].sum() / max(1, sw[y == 1].sum())), sw)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                       l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=bal)
    Xa = b.set_index("bid")[FEATS[:-2]].copy()
    Xa[["hist_own", "hist_block"]] = history(D, b, 2100).loc[Xa.index].to_numpy()
    p = np.clip(m.predict_proba(Xa)[:, 1], 1e-4, 1 - 1e-4)
    theta = 100 * (z_high - coef["g0"] - coef["g2_logit_pstatic"] * np.log(p / (1 - p))) / coef["g1_score"]
    gid = pd.read_parquet(sd.OUT / "buildings.parquet", columns=["bid", "gid"]).set_index("bid").gid
    out = pd.DataFrame({"gid": gid.loc[Xa.index].to_numpy(), "p_static": p,
                        "theta_high": np.clip(theta, 10, 95), "delta": np.clip(theta, 10, 95) - 40})
    out.to_parquet(OUT / out_name)
    meta = {"g0": coef["g0"], "g1": coef["g1_score"], "g2": coef["g2_logit_pstatic"], "z_high": z_high,
            "trained_on": "2010~2022 사건 (보정 기간)", "history_up_to": 2025, "features": FEATS}
    (OUT / out_name.replace(".parquet", "_meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("정적 취약성 %d동: Δ 중앙값 %.1f, 10~90%% [%.1f, %.1f]", len(out), out.delta.median(),
             out.delta.quantile(.1), out.delta.quantile(.9))
    return out
