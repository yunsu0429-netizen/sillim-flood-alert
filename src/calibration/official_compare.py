"""서울시 공식 침수예보 기준과 우리 모델 비교 (시험 기간 2023~2025, 건물 단위).

비교 대상
  official23  서울시 침수예보(2023-05~, 자치구 단위): 시간당 강우 55mm 초과 또는 15분 강우 20mm 초과
              (도로수위계 15cm 조건은 과거 자료가 없어 제외 → 공식 체계의 하한 근사)
  official26  2026 기준: 시간당 72mm 초과, 또는 시간당 50mm 이상 + 3시간 90mm 이상
  v02         운영 모델 v0.2 (SWMM 근사식 + 회귀 가중치 + 건물 정적 취약성 경계 Δ_i)
  exact       v0.2와 같되 지표면을 SWMM 5 원식(소영역별 α 식 3-8·3-9, PctZero, 적분형 Horton+회복 §4.2.3)으로

강우는 기상청 AWS 10분 자료를 건물 위치로 IDW 보간한 값(fit_seoul.Prepared). 공식 기준은 자치구 안 어느 한 곳이라도
넘으면 그 구 전체를 발령하므로, 같은 사건·같은 구 표본 건물 중 가장 이른 초과 시각을 구 발령 시각으로 쓴다.
15분 강우는 10분 칸 안 강우가 고르다고 보고 max(P10[t] + ½·P10[t+1], ½·P10[t−1] + P10[t])로 근사.

실행: .venv\\Scripts\\python -m src.calibration.official_compare rules exact compare
"""
import argparse
import copy
import json
import logging

import numpy as np
import pandas as pd

from src.calibration import fit_seoul as fs
from src.calibration import improve as im
from src.calibration import seoul_data as sd
from src.calibration.simulate import window_sum

log = logging.getLogger(__name__)
OUT = im.OUT


# ---------------------------------------------------------------- 공식 기준 발령 시각
def rule_times(p) -> pd.DataFrame:
    p10 = np.nan_to_num(p.P10.astype(np.float32))
    p60 = window_sum(p10, 6)
    p180 = window_sum(p10, 18)
    tmin = np.array([(pd.Timestamp(t) - im.EPOCH).total_seconds() / 60 for t in p.times])
    cont = np.concatenate([[False], np.diff(tmin) <= 10])          # 앞 칸과 10분 연속인지 (마른 구간은 생략돼 있음)
    nxt = np.pad(p10[:, 1:] * cont[None, 1:], ((0, 0), (0, 1)))
    prv = np.pad(p10[:, :-1] * cont[None, 1:], ((0, 0), (1, 0)))
    s15 = np.maximum(p10 + 0.5 * nxt, 0.5 * prv + p10)

    def first(mask):
        return np.where(mask.any(axis=1), tmin[mask.argmax(axis=1)], np.nan)
    return pd.DataFrame({"bid": p.bids, "event": p.ev.eid,
                         "t_off23": first((p60 > 55) | (s15 > 20)),
                         "t_off26": first((p60 > 72) | ((p60 >= 50) & (p180 >= 90))),
                         "max_p60": p60.max(axis=1), "max_s15": s15.max(axis=1)})


def to_district(r: pd.DataFrame) -> pd.DataFrame:
    sgg = pd.read_parquet(sd.OUT / "buildings.parquet", columns=["bid", "sgg"]).set_index("bid").sgg
    r = r.assign(sgg=sgg.reindex(r.bid).to_numpy())
    g = r.groupby(["event", "sgg"])[["t_off23", "t_off26"]].min().add_suffix("_gu")
    return r.merge(g, left_on=["event", "sgg"], right_index=True, how="left")


# ---------------------------------------------------------------- Jang (2015) 자치구별 다중 임계값
# Jang, J.-H. (2015) Water 7(11):6056–6078, doi:10.3390/w7116056
#   §2.3 식 (1): 비침수 사건 관측값 RH⁰ᵢ > RTᵢ 이면 RTᵢ ← RH⁰ᵢ, 침수 사건 RH¹ᵢ < RTᵢ 이면 RTᵢ ← RH¹ᵢ (사건 순서대로)
#   §2.4·그림 8: 배수 임계(짧은 지속시간) 초과 → 주황, 유출 임계(긴 지속시간)만 → 노랑, 둘 다 → 빨강
#   §3.1 식 (2)–(4): 적중률 h/(h+m), 오경보율 f/(f+c), 정확도 (h+c)/(h+m+f+c)
# 서울 적용: 지속시간 10분·1시간·3시간(배수) / 12시간(유출). 초기값(그림 8의 n=1 입력)은 침수모형 대신 공식 기준:
#   1h 55mm·15분 20mm(→10분 13.3mm, 칸 안 균등 가정)·3h 90mm(서울 2026 기준) / 12h 110mm(기상청 호우주의보)
JANG_DUR = {10: 1, 60: 6, 180: 18, 720: 72}          # 분 → 10분 칸
JANG_DRAIN, JANG_RUNOFF = (10, 60, 180), (720,)
JANG_INIT = {10: 20 * 10 / 15, 60: 55.0, 180: 90.0, 720: 110.0}


def regular_sums(p, durs: dict) -> tuple[np.ndarray, dict]:
    """마른 구간을 뺀 Prepared 시각을 10분 등간격으로 복원해 이동합 → 원래 시각만 반환."""
    p10 = np.nan_to_num(p.P10.astype(np.float32))
    tmin = np.array([(pd.Timestamp(t) - im.EPOCH).total_seconds() / 60 for t in p.times])
    idx = np.round((tmin - tmin[0]) / 10).astype(int)
    full = np.zeros((p10.shape[0], idx[-1] + 1), dtype=np.float32)
    full[:, idx] = p10
    return tmin, {d: window_sum(full, k)[:, idx] for d, k in durs.items()}


def jang_run(preps, freeze_year: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """사건 순서대로 자치구 임계값을 갱신하며(식 1) 발령 시각을 계산. freeze_year 이후엔 갱신 중지(엄격 시험용)."""
    b = pd.read_parquet(sd.OUT / "buildings.parquet", columns=["bid", "sgg"]).set_index("bid").sgg
    lab = pd.read_parquet(sd.OUT / "labels.parquet")
    flooded = lab[lab.flood == 1].assign(sgg=lambda x: b.reindex(x.bid).to_numpy()).groupby("event").sgg.unique()
    RT: dict = {}
    rows, gu_rows = [], []
    for p in sorted(preps, key=lambda q: q.ev.days[0]):
        year = int(p.ev.days[0][:4])
        tmin, S = regular_sums(p, JANG_DUR)
        sgg = b.reindex(p.bids).to_numpy()
        fl = set(flooded.get(p.ev.eid, []))
        t_bldg = np.full(len(p.bids), np.nan)
        lvl_bldg = np.zeros(len(p.bids), dtype=int)
        for g in np.unique(sgg):
            m = sgg == g
            G = {d: S[d][m].max(axis=0) for d in JANG_DUR}                     # 구 안 어느 한 곳이라도 (서울 방식)
            rt = RT.setdefault(g, dict(JANG_INIT))
            ex = {d: G[d] > rt[d] for d in JANG_DUR}
            drain = np.any([ex[d] for d in JANG_DRAIN], axis=0)
            runoff = np.any([ex[d] for d in JANG_RUNOFF], axis=0)
            light = np.where(drain & runoff, 3, np.where(drain, 2, np.where(runoff, 1, 0)))  # 노1·주황2·빨강3
            first = tmin[light.argmax()] if light.any() else np.nan
            t_bldg[m], lvl_bldg[m] = first, light.max()
            RH = {d: float(G[d].max()) for d in JANG_DUR}
            gu_rows.append({"event": p.ev.eid, "year": year, "sgg": g, "flood": int(g in fl), "warned": bool(light.any()),
                            "max_light": int(light.max()), "t_warn": first, **{f"RT{d}": rt[d] for d in JANG_DUR},
                            **{f"RH{d}": RH[d] for d in JANG_DUR}})
            if freeze_year is None or year < freeze_year:                       # 식 (1)
                for d in JANG_DUR:
                    if g in fl and RH[d] < rt[d]:
                        rt[d] = RH[d]
                    elif g not in fl and RH[d] > rt[d]:
                        rt[d] = RH[d]
        rows.append(pd.DataFrame({"bid": p.bids, "event": p.ev.eid, "t_jang": t_bldg, "jang_light": lvl_bldg}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(gu_rows), RT


def gu_metrics(g: pd.DataFrame, warned_col="warned") -> dict:
    """Jang (2015) 식 (2)–(4), 단위 = 자치구×사건."""
    w, y = g[warned_col].astype(bool), g.flood.astype(bool)
    h, m, f, c = int((w & y).sum()), int((~w & y).sum()), int((w & ~y).sum()), int((~w & ~y).sum())
    return {"h": h, "m": m, "f": f, "c": c, "hit_rate": round(h / max(h + m, 1), 4),
            "false_alarm_rate": round(f / max(f + c, 1), 4), "overall_correct": round((h + c) / max(h + m + f + c, 1), 4),
            "CSI": round(h / max(h + m + f, 1), 4)}


# ---------------------------------------------------------------- 건물별 경계(정적 취약성) 경보 시각
def per_building_theta(d: pd.DataFrame, coef: dict) -> tuple[np.ndarray, float]:
    """θ_i = 100·(z* − g0 − g2·logit p_i)/g1, z*는 보정 기간 경보 비율이 θ=40과 같아지는 값 (improve.export_static)."""
    lp = np.log(d.p_static / (1 - d.p_static)).to_numpy()
    tr = d.train.to_numpy()
    w = d.weight.to_numpy()
    target = np.average(im.warn_time(d[tr], 40).notna(), weights=w[tr])
    best = None
    for z in np.arange(-4.0, 1.0, 0.05):
        th = np.clip(100 * (z - coef["g0"] - coef["g2_logit_pstatic"] * lp) / coef["g1_score"], 10, 95)
        share = np.average(warn_at(d[tr], th[tr]).notna(), weights=w[tr])
        if best is None or abs(share - target) < best[0]:
            best = (abs(share - target), z, th)
    return best[2], float(best[1])


def warn_at(d: pd.DataFrame, theta) -> pd.Series:
    """건물마다 다른 경계 θ_i (5점 격자로 반올림)로 경보 시각."""
    th = np.clip(np.round(np.asarray(theta) / 5) * 5, 10, 95).astype(int)
    T = d[[f"t_{k}" for k in im.THETAS]].to_numpy()
    t_score = T[np.arange(len(d)), (th - 10) // 5]
    return pd.Series(np.fmin(t_score, d[["t_excess", "t_override"]].min(axis=1).to_numpy()), index=d.index)


def metrics(d: pd.DataFrame, t_warn: pd.Series) -> dict:
    w, y = d.weight.to_numpy(), d.flood.to_numpy()
    warn = t_warn.notna().to_numpy()
    H = w[warn & (y == 1)].sum(); M = w[~warn & (y == 1)].sum(); F = w[warn & (y == 0)].sum()
    out = {"POD": H / (H + M), "FAR": F / (H + F) if H + F else np.nan, "CSI": H / (H + M + F),
           "warned_share": float(np.average(warn, weights=w)), "false_alarm_bldg_per_hit": F / H if H else np.nan}
    f = d.flood.eq(1) & d.t_damage.notna()
    lead = (d.t_damage[f] - t_warn[f])
    out.update({"timed": int(f.sum()), "warned_before_damage": float((lead > 0).sum() / f.sum()),
                "lead_median_min": float(lead.dropna().median()) if lead.notna().any() else np.nan})
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def matched_budget(d: pd.DataFrame, share: float, theta_i: np.ndarray) -> tuple[pd.Series, float]:
    """경보 비율(가중)이 share와 같아지도록 θ_i 전체를 같은 폭 s만큼 이동."""
    w = d.weight.to_numpy()
    best = None
    for s in range(-60, 61, 5):
        t = warn_at(d, theta_i + s)
        diff = abs(np.average(t.notna(), weights=w) - share)
        if best is None or diff < best[0]:
            best = (diff, s, t)
    return best[2], best[1]



# ---------------------------------------------------------------- 공식 기준: 서울시 강우량계 그대로 (자치구별)
def gauge_rule_times() -> pd.DataFrame:
    """서울시 강우량계(OA-22715) 자치구별 발령 시각. 구 안 강우계 중 하나라도 기준을 넘는 첫 시각(구간 끝, epoch 분)."""
    from src.calibration import seoul_gauges as sg
    st = sg.stations().set_index("stn")
    rows = []
    for _, e in sd.load_events().iterrows():
        parts = [pd.read_csv(sg.OUT / f"{d}.csv") for d in e.days.split() if (sg.OUT / f"{d}.csv").exists()]
        if not parts:
            continue
        a = pd.concat(parts)
        a["ts"] = pd.to_datetime(a.ts)
        for stn, g in a.groupby("stn"):
            s = g.set_index("ts").rn10.sort_index()
            s = s[~s.index.duplicated()].asfreq("10min").fillna(0.0)
            p60, p180 = s.rolling(6, min_periods=1).sum(), s.rolling(18, min_periods=1).sum()
            s15 = np.maximum(s + 0.5 * s.shift(-1, fill_value=0), 0.5 * s.shift(1, fill_value=0) + s)
            tm = (s.index - im.EPOCH).total_seconds() / 60

            def first(m):
                m = np.asarray(m)
                return float(tm[m.argmax()]) if m.any() else np.nan
            rows.append({"event": e.event, "gu": st.gu.get(stn), "stn": stn,
                         "t_g23": first((p60 > 55) | (s15 > 20)), "t_g26": first((p60 >= 72) | ((p60 >= 50) & (p180 >= 90)))})
    r = pd.DataFrame(rows)
    return r.groupby(["event", "gu"], as_index=False)[["t_g23", "t_g26"]].min()


def attach_gauge_rules(d: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    from src.calibration import static_extra
    nm = pd.read_parquet(static_extra.OUT, columns=["bid", "sgg_name"]).set_index("bid").sgg_name
    d = d.assign(gu=nm.reindex(d.bid).to_numpy())
    return d.merge(g, on=["event", "gu"], how="left")


def curve(dt: pd.DataFrame, tht: np.ndarray, gate: str | None = None, mode: str = "and") -> list:
    """경계 이동 폭별 (경보 비율, POD, CSI, 피해 전 경보). gate: 공식 기준 발령 시각 열과 AND/OR 결합."""
    out = []
    for sft in range(-40, 61, 5):
        t = warn_at(dt, tht + sft)
        if gate:
            g = dt[gate]
            if mode == "and":     # 두 조건 모두: 늦은 쪽 시각, 하나라도 없으면 경보 없음
                t = pd.Series(np.where(t.notna() & g.notna(), np.fmax(t, g), np.nan), index=dt.index)
            else:                 # 하나라도: 이른 쪽 시각
                t = pd.Series(np.fmin(t, g), index=dt.index)
        out.append({"shift": sft, **metrics(dt, t)})
    return out


def at_budget(cur: list, share: float) -> dict:
    """곡선에서 경보 비율이 share 이하인 점 중 POD 최대 (같은 경보 규모 이하에서 비교)."""
    ok = [c for c in cur if c["warned_share"] <= share * 1.02]
    return max(ok, key=lambda c: (c["POD"], c["warned_before_damage"])) if ok else {}


# ---------------------------------------------------------------- main
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["rules", "jang", "exact", "compare", "gauge_rules", "extra", "dense", "compare2"])
    a = ap.parse_args()
    rep_path = OUT / "official_compare.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
    save = lambda: rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    need = {"rules", "jang", "exact", "dense"} & set(a.steps)
    D = fs.Data() if need else None
    preps = [fs.Prepared(D, e) for _, e in D.events.iterrows()] if need else None
    if "rules" in a.steps:
        r = to_district(pd.concat([rule_times(p) for p in preps], ignore_index=True))
        r.to_parquet(OUT / "table_rules.parquet")
        log.info("공식 기준 시각 %d행", len(r))
    if "jang" in a.steps:
        jr = {}
        for tag, fz in (("online", None), ("frozen2022", fs.SPLIT_YEAR)):
            jb, jg, RT = jang_run(preps, fz)
            jb.rename(columns={"t_jang": f"t_jang_{tag}", "jang_light": f"jang_light_{tag}"}).to_parquet(
                OUT / f"table_jang_{tag}.parquet")
            jg.to_csv(OUT / f"jang_gu_{tag}.csv", index=False, encoding="utf-8-sig")
            te = jg[jg.year >= fs.SPLIT_YEAR]
            jr[tag] = {"gu_test": gu_metrics(te), "gu_train": gu_metrics(jg[jg.year < fs.SPLIT_YEAR]),
                       "final_thresholds_median": {d: round(float(np.median([v[d] for v in RT.values()])), 1)
                                                   for d in JANG_DUR}}
        # 같은 단위(자치구×사건)로 공식 기준도
        rules = pd.read_parquet(OUT / "table_rules.parquet")
        og = rules.groupby(["event", "sgg"]).t_off23_gu.min().notna().rename("warned").reset_index()
        og = og.merge(jg[["event", "sgg", "flood", "year"]], on=["event", "sgg"])
        jr["official23_gu_test"] = gu_metrics(og[og.year >= fs.SPLIT_YEAR])
        rep["jang"] = jr
        log.info("jang %s", jr); save()
    if "exact" in a.steps:
        rm = im.seoul_model()
        rm["surface"]["swmm"]["exact"] = True
        d1 = im.build(D, preps, rm)
        w, wi = fs.fit_weights(d1, rm["weights"])                     # 같은 방법(로지스틱 회귀)으로 가중치 재추정
        rm["weights"] = w
        d1 = im.build(D, preps, rm)
        d1, info = im.step_D(D, d1)
        d1.to_parquet(OUT / "table_exact.parquet")
        rep["exact"] = {"weights": w, "weights_fit": wi, "D": info, "score": im.score_metrics(d1, "max_score"),
                        "combined": im.score_metrics(d1, "P_comb_raw")}
        log.info("exact %s", rep["exact"]); save()
    if "gauge_rules" in a.steps:
        g = gauge_rule_times()
        g.to_parquet(OUT / "table_gauge_rules.parquet")
        log.info("강우량계 공식 기준 %d (사건×구)", len(g))
    if "extra" in a.steps:                                            # AWS 강우 그대로 + 공공데이터 정적 특성
        assert im.EXTRA, "FLOOD_EXTRA_FEATS=1 필요"
        D = D or fs.Data()
        d1, info = im.step_D(D, pd.read_parquet(OUT / "table_base.parquet"))
        d1.to_parquet(OUT / "table_D_extra.parquet")
        rep["extra"] = {"D": info, "combined": im.score_metrics(d1, "P_comb_raw")}
        log.info("extra %s", rep["extra"]); save()
    if "dense" in a.steps:                                            # 서울시 강우량계 추가 + 정적 특성
        assert fs.RAIN_NET == "aws+seoul" and im.EXTRA, "FLOOD_RAIN_NET=aws+seoul, FLOOD_EXTRA_FEATS=1 필요"
        rm = im.seoul_model()
        d1 = im.build(D, preps, rm)
        w, wi = fs.fit_weights(d1, rm["weights"])
        rm["weights"] = w
        d1 = im.build(D, preps, rm)
        d1.to_parquet(OUT / "table_dense_base.parquet")
        d2, info = im.step_D(D, d1)
        d2.to_parquet(OUT / "table_dense.parquet")
        rep["dense"] = {"weights": w, "weights_fit": wi, "D": info, "score": im.score_metrics(d1, "max_score"),
                        "combined": im.score_metrics(d2, "P_comb_raw")}
        log.info("dense %s", rep["dense"]); save()
    if "compare2" in a.steps:
        g = pd.read_parquet(OUT / "table_gauge_rules.parquet")
        coef_base = json.loads((OUT / "D_F_on_base.json").read_text(encoding="utf-8"))["D_on_base"]["combine_coef"]
        variants = {"v02": (OUT / "table_D_on_base.parquet", coef_base)}
        for k, f in (("v02_extra", "table_D_extra.parquet"), ("v03_dense", "table_dense.parquet")):
            if (OUT / f).exists():
                key = {"v02_extra": "extra", "v03_dense": "dense"}[k]
                variants[k] = (OUT / f, rep[key]["D"]["combine_coef"])
        res = {"official": {}, "variants": {}}
        for name, (f, coef) in variants.items():
            d = attach_gauge_rules(pd.read_parquet(f), g)
            jf = OUT / "table_jang_online.parquet"
            if jf.exists():
                d = d.merge(pd.read_parquet(jf), on=["bid", "event"], how="left")
            th, z = per_building_theta(d, coef)
            te = ~d.train.to_numpy()
            dt, tht = d[te], th[te]
            if not res["official"]:
                res["official"] = {"seoul23_gauge": metrics(dt, dt.t_g23), "seoul26_gauge": metrics(dt, dt.t_g26)}
                if "t_jang_online" in dt:
                    res["official"]["jang2015_online"] = metrics(dt, dt.t_jang_online)
                res["n_test"] = {"rows": int(len(dt)), "flooded": int(dt.flood.sum()), "events": int(dt.event.nunique())}
            v = {"z_star": round(z, 2), "operating": metrics(dt, warn_at(dt, tht))}
            curves = {"model": curve(dt, tht), "and23": curve(dt, tht, "t_g23", "and"),
                      "and26": curve(dt, tht, "t_g26", "and"), "or26": curve(dt, tht, "t_g26", "or")}
            if "t_jang_online" in dt:
                curves["andJang"] = curve(dt, tht, "t_jang_online", "and")
            for ok in res["official"]:
                share = res["official"][ok]["warned_share"]
                v[f"budget_{ok}"] = {c: at_budget(cv, share) for c, cv in curves.items()}
            v["curves"] = curves
            res["variants"][name] = v
        rep["compare2"] = res
        save()
        brief = {k: {kk: vv for kk, vv in v.items() if kk != "curves"} for k, v in res["variants"].items()}
        log.info("compare2 official %s | %s", res["official"], json.dumps(brief, ensure_ascii=False, indent=1))
    if "compare" in a.steps:
        rules = pd.read_parquet(OUT / "table_rules.parquet")
        coef_base = json.loads((OUT / "D_F_on_base.json").read_text(encoding="utf-8"))["D_on_base"]["combine_coef"]
        res = {}
        variants = {"v02": (pd.read_parquet(OUT / "table_D_on_base.parquet"), coef_base)}
        if (OUT / "table_exact.parquet").exists():
            variants["exact"] = (pd.read_parquet(OUT / "table_exact.parquet"), rep["exact"]["D"]["combine_coef"])
        for name, (d, coef) in variants.items():
            d = d.merge(rules.drop(columns=["sgg"]), on=["bid", "event"], how="left")
            for tag in ("online", "frozen2022"):
                f = OUT / f"table_jang_{tag}.parquet"
                if f.exists():
                    d = d.merge(pd.read_parquet(f), on=["bid", "event"], how="left")
            th, z = per_building_theta(d, coef)
            te = ~d.train.to_numpy()
            dt, tht = d[te], th[te]
            if "official" not in res:
                res["official"] = {k: metrics(dt, dt[c]) for k, c in
                                   [("official23_gu", "t_off23_gu"), ("official23_bldg", "t_off23"),
                                    ("official26_gu", "t_off26_gu"), ("official26_bldg", "t_off26")]
                                   + [(f"jang_{g}_gu", f"t_jang_{g}") for g in ("online", "frozen2022")
                                      if f"t_jang_{g}" in dt]}
                res["n_test"] = {"rows": int(len(dt)), "flooded": int(dt.flood.sum()), "events": int(dt.event.nunique())}
            out = {"z_star": round(z, 2), "operating": metrics(dt, warn_at(dt, tht)),
                   "score_only_theta40": metrics(dt, im.warn_time(dt, 40))}
            for k in [k for k in res["official"] if k.endswith("_gu")]:
                share = res["official"][k]["warned_share"]
                t, s = matched_budget(dt, share, tht)
                out[f"same_budget_as_{k}"] = {"shift": s, **metrics(dt, t)}
            res[name] = out
        rep["compare"] = res
        log.info("compare %s", json.dumps(res, ensure_ascii=False, indent=1)); save()


if __name__ == "__main__":
    main()
