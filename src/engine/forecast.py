"""건물별 위험도 계산 엔진 (현재 + 10~60분 예보). 수식은 src/engine/risk_model.py."""
import csv
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pyproj

from src.config import PROCESSED
from src.engine.risk_model import (LEVELS_KO, Context, RiskModel, official_rule, predict_sewer, predict_sewer_arx,
                                   rain_features, sewer_features)
from src.sources.sewer_api import REPLAY_CSV

_TO5186 = pyproj.Transformer.from_crs(4326, 5186, always_xy=True)
EVENT_MAX_LABEL = "2022-08-08 최고수위"


@lru_cache(maxsize=1)
def event_max_levels() -> dict:
    """2022-08-08 수위계별 최고수위 — 관 용량(H_critical) 자료가 없어 '만관' 기준으로 가정."""
    out = {}
    if REPLAY_CSV.exists():
        with open(REPLAY_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["station"]] = max(out.get(r["station"], 0.0), float(r["level_m"]))
    return out


@dataclass
class Snapshot:
    t: datetime
    ctx: Context
    scores: np.ndarray        # 현재(h=0) 점수, 반지하 아니면 NaN
    grades: np.ndarray        # 화면 등급 (히스테리시스 적용), -1 = 비대상/판정 불가
    levels: np.ndarray        # [시점, 건물] 예보 등급 (히스테리시스 없음)
    t_flood: np.ndarray       # 개구부 유입 예상(분), -1 없음
    t_protected: np.ndarray   # 차수판 고려 유입 예상(분)
    confidence: np.ndarray    # 0~100

    @property
    def rain_now(self):
        return self.ctx.rain[0]


class RiskEngine:
    def __init__(self, params: dict, static: gpd.GeoDataFrame | None = None, stations: list | None = None):
        self.p = params
        self.cfg = params["risk_model"]
        self.static = static if static is not None else gpd.read_parquet(PROCESSED / "buildings_static.parquet")
        self.static = self.static.sort_values("bid").reset_index(drop=True)
        self.banjiha = self.static["is_banjiha"].to_numpy()
        self.stations = stations or []
        idx, dist = self._link_stations()
        self.st_idx, self.st_dist = idx, dist
        self.h_crit, self.h_crit_src = self._h_critical()
        self._arx = self._load_arx()
        self.model = RiskModel(self.cfg, self.static, idx, dist, self._dem_quality())
        self.vulnerable = np.zeros(len(self.static), dtype=bool)
        self.p_static = np.full(len(self.static), np.nan)
        self.model.delta = self._load_static_prior()
        self.horizons = list(self.cfg["horizons_min"])
        self._cur = np.full(len(self.static), -1)
        self._below = np.full(len(self.static), np.nan)

    # ---------------- 준비
    def _link_stations(self):
        n = len(self.static)
        if not self.stations:
            return np.full(n, -1), np.full(n, np.nan)
        sx, sy = _TO5186.transform([s["lng"] for s in self.stations], [s["lat"] for s in self.stations])
        pts = self.static.geometry.representative_point()
        bx, by = pts.x.to_numpy(), pts.y.to_numpy()
        d = np.hypot(bx[:, None] - np.array(sx)[None, :], by[:, None] - np.array(sy)[None, :])
        j = d.argmin(axis=1)
        dist = d[np.arange(n), j]
        return np.where(dist <= self.cfg["sewer"]["station_radius_m"], j, -1), dist

    def _h_critical(self):
        ov = self.cfg["sewer"].get("h_critical_override") or {}
        ev = event_max_levels()
        known = [v for v in ev.values() if v > 0]
        med = statistics.median(known) if known else 1.5
        vals, src = [], []
        for s in self.stations:
            sid = s["station"]
            if sid in ov:
                vals.append(float(ov[sid])), src.append("override")
            elif ev.get(sid, 0) > 0:
                vals.append(ev[sid]), src.append(EVENT_MAX_LABEL)
            else:
                vals.append(med), src.append("중앙값 대체")
        return np.array(vals), src

    def _load_static_prior(self) -> np.ndarray:
        """D. 건물 정적 취약성 → 건물별 경계 조정 Δ (src/calibration/improve.export_static). 없는 건물은 0."""
        import pandas as pd
        from src.config import ROOT
        sp = self.cfg.get("static_prior") or {}
        path = ROOT / sp.get("path", "data/calibration/seoul/improve/static_prior.parquet")
        if not sp.get("enabled", False) or not path.exists() or "gid" not in self.static:
            return np.zeros(len(self.static))
        d = pd.read_parquet(path, columns=["gid", "delta", "p_static"]).drop_duplicates("gid").set_index("gid")
        lim = float(sp.get("max_abs_delta", 30))
        # 평상시 취약 건물: 반지하 후보 중 정적 취약성 상위 top_pct% → 지도 테두리 (등급과 별개)
        p = self.static.gid.map(d.p_static).to_numpy(dtype=float)
        top = float((self.cfg.get("vulnerable") or {}).get("top_pct", 10))
        ok = self.banjiha & ~np.isnan(p)
        cut = np.quantile(p[ok], 1 - top / 100) if ok.any() else np.inf
        self.vulnerable = ok & (np.nan_to_num(p, nan=-1) >= cut)
        self.p_static = p
        return np.clip(self.static.gid.map(d.delta).fillna(0).to_numpy(dtype=float), -lim, lim)

    def _load_arx(self) -> dict:
        """C. 수위계별 ARX 계수 (src/calibration/sewer_forecast.py 결과). 없으면 빈 dict → 지속 예측."""
        import json
        from src.config import ROOT
        path = ROOT / self.cfg["sewer"].get("arx_coef", "data/calibration/seoul/improve/C_arx_coef.json")
        try:
            return {k: np.asarray(v, dtype=float) for k, v in json.load(open(path, encoding="utf-8")).items()}
        except (OSError, ValueError):
            return {}

    def _dem_quality(self):
        q = self.cfg["confidence"]["elevation_quality"]
        try:
            import json
            res = json.load(open(PROCESSED / "buildings_3d.json", encoding="utf-8"))["meta"]["sources"]["dem"]
            return q["dem5"] if "5 m" in res or "5m" in res else q["dem90"]
        except Exception:
            return q["dem90"]

    def set_openings(self, rows: list[dict]):
        """DB 개구부 행 → 건물별 가장 낮은 문턱. 없는 건물은 입력된 건물들의 평균으로 대체."""
        per = {}
        for r in rows:
            if r.get("sill_above_ground_m") is None or r["bid"] < 0 or r["bid"] >= len(self.static):
                continue
            b = r["bid"]
            cur = per.get(b)
            if cur is None or r["sill_above_ground_m"] < cur["sill"]:
                per[b] = {"sill": r["sill_above_ground_m"], "barrier": r.get("barrier_m") or 0.0,
                          "quality": r.get("quality") or "manual", "source": r.get("source")}
        imputed = None
        if per:
            imputed = {"sill": float(np.mean([v["sill"] for v in per.values()])), "n": len(per)}
        self.model.set_openings(per, imputed)

    # ---------------- 판정
    def district_gate(self, t: datetime, gauge_obs: dict | None, rain_forecast) -> np.ndarray | None:
        """자치구 강우량계 중 하나라도 서울시 침수예보 기준을 (t+h − lookback, t+h] 안에 충족했는지 [시점].
        관측은 강우계별, t 이후는 기상청 초단기예보(구 공통)를 더한다. 강우계 자료가 없으면 None(게이트 미적용)."""
        dg = self.cfg.get("district_gate") or {}
        if not dg.get("enabled") or not gauge_obs:
            return None
        look = int(dg.get("lookback_min", 180))
        d10 = timedelta(minutes=10)
        ks = list(range(-(look + 180) // 10 + 1, self.horizons[-1] // 10 + 1))
        times = [t + d10 * k for k in ks]
        fc = sorted(rain_forecast or [])

        def mm(series, a, b):
            return sum(v for ts, v in series if a < ts <= b)
        rows = []
        for series in gauge_obs.values():
            obs = sorted(series)
            if not any(ts > t - timedelta(minutes=60) for ts, _ in obs):
                continue                                     # 최근 자료 없는 강우계 제외
            future = any(ts > t for ts, _ in obs)           # 재생: 이후 실측이 곧 "완벽 예보"
            rows.append([mm(obs if (ts <= t or future) else fc, ts - d10, ts) for ts in times])
        if not rows:
            return None
        hit = official_rule(np.array(rows), dg.get("rule", "seoul2023")).any(axis=0)
        out = []
        for h in self.horizons:
            end = ks.index(h // 10)
            out.append(bool(hit[max(0, end - look // 10 + 1): end + 1].any()))
        return np.array(out)

    def context(self, t: datetime, rain_obs, rain_forecast, sewer_obs: dict, rain_quality: float,
                live: bool, gauge_obs: dict | None = None) -> Context:
        rain = {}
        rain[0] = rain_features(rain_obs, t)
        if live and rain[0] is not None:
            steps = sum(1 for ts, _ in rain_obs if t - timedelta(minutes=60) < ts <= t)
            last = max((ts for ts, _ in rain_obs if ts <= t), default=None)
            if steps < 4 or last is None or (t - last).total_seconds() > 30 * 60:
                rain[0] = None                         # 자료 부족·지연 → 0으로 보지 않고 결측
            rain_quality = 0 if rain[0] is None else min(100, steps / 6 * 100)
        for h in self.horizons[1:]:
            rain[h] = rain_features(rain_forecast, t + timedelta(minutes=h)) if rain_forecast else None
        feats = [sewer_features(sewer_obs.get(s["station"], []), t, self.cfg["sewer"]["stale_minutes"])
                 for s in self.stations]
        cap = self.cfg["sewer"]["pred_cap_ratio"]
        hp = np.full((len(self.horizons), len(self.stations)), np.nan)
        method = self.cfg["sewer"].get("predictor", "linear")
        if method == "arx":
            def mm(series, a, b):
                return sum(v for ts, v in (series or []) if a < ts <= b)
            d10 = timedelta(minutes=10)
            past = [mm(rain_obs, t - d10 * (i + 1), t - d10 * i) for i in range(5, -1, -1)]
            steps = self.horizons[-1] // 10
            fut = [mm(rain_forecast, t + d10 * i, t + d10 * (i + 1)) for i in range(steps)]
        for j, f in enumerate(feats):
            if f is None:
                continue
            coef = self._arx.get(self.stations[j]["station"], self._arx.get("_pooled")) if method == "arx" else None
            if coef is not None:
                pred = predict_sewer_arx(coef, f["H0"], f["H10"] if f["H10"] is not None else f["H0"], past, fut, steps,
                                         self.h_crit[j], cap)
                for k, h in enumerate(self.horizons):
                    hp[k, j] = f["H0"] if h == 0 else pred[h // 10 - 1]
            else:
                for k, h in enumerate(self.horizons):     # 계수 없음: 지속(현재 수위 유지)이 직선 외삽보다 정확 (C 시험)
                    hp[k, j] = (predict_sewer(f, h, self.h_crit[j], cap) if method == "linear"
                                else min(f["H0"], cap * self.h_crit[j]))
        util = hp / self.h_crit[None, :] if len(self.stations) else hp
        ss = (self._surface_series(t, rain_obs, rain_forecast, sewer_obs, feats, hp)
              if self.cfg["surface"].get("method") == "swmm" else None)
        return Context(t, self.horizons, rain, float(rain_quality), [s["station"] for s in self.stations],
                       feats, self.h_crit, self.h_crit_src, hp, util, surface_series=ss,
                       gate=self.district_gate(t, gauge_obs, rain_forecast))

    def _surface_series(self, t, rain_obs, rain_forecast, sewer_obs, feats, hp):
        """SWMM 적분용 10분 시계열: t−warmup ~ t+60. 과거는 관측 강우·실측 수위, 미래는 예보 강우·예측 수위."""
        warm = int(self.cfg["surface"].get("warmup_min", 360))
        ks = list(range(-warm // 10 + 1, self.horizons[-1] // 10 + 1))
        times = [t + timedelta(minutes=10 * k) for k in ks]
        obs = sorted(rain_obs or [])
        fc = sorted(rain_forecast or [])

        def mm(series, a, b):
            return sum(v for ts, v in series if a < ts <= b)
        rain = np.array([mm(obs if ts <= t else fc, ts - timedelta(minutes=10), ts) for ts in times])
        util = np.full((len(times), len(self.stations)), np.nan)
        hz = np.array(self.horizons)
        for j, s in enumerate(self.stations):
            if feats[j] is None:
                continue
            rows = sorted((ts.timestamp(), lv) for ts, lv, *_ in sewer_obs.get(s["station"], []) if ts <= t)
            if rows:
                rt, rl = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
            for i, ts in enumerate(times):
                if ts > t:
                    util[i, j] = np.interp((ts - t).total_seconds() / 60, hz, hp[:, j]) / self.h_crit[j]
                elif rows:
                    x = ts.timestamp()
                    q = np.searchsorted(rt, x, side="right") - 1          # 그 시각 이전 15분 안 마지막 수위
                    if q >= 0 and x - rt[q] < 900:
                        util[i, j] = rl[q] / self.h_crit[j]
        h_index = [ks.index(h // 10) for h in self.horizons]
        return {"rain_mm": rain, "dt_min": np.full(len(times), 10.0), "util": util, "h_index": h_index,
                "times": times}

    def compute(self, t: datetime, rain_obs, rain_forecast, sewer_obs: dict, rain_quality: float,
                live: bool, gauge_obs: dict | None = None) -> Snapshot:
        ctx = self.context(t, rain_obs, rain_forecast, sewer_obs, rain_quality, live, gauge_obs)
        r = self.model.compute(ctx)
        grades = self._hysteresis(r["level"][0], r["score"][0], t)
        return Snapshot(t, ctx, r["score"][0], grades, r["level"].astype(np.int8), r["t_flood"].astype(np.int16),
                        r["t_protected"].astype(np.int16), r["confidence"].astype(np.int16))

    def _hysteresis(self, raw, score, t):
        """상승은 즉시, 하강은 (현재 등급 하한 − hysteresis_points) 미만이 hysteresis_minutes 지속될 때 한 단계."""
        now = t.timestamp() / 60
        pts, mins = self.p["score"]["hysteresis_points"], self.p["score"]["hysteresis_minutes"]
        th = np.array([0] + list(self.cfg["levels"]))
        cur, below = self._cur, self._below
        up = raw >= cur
        cur = np.where(up, raw, cur)
        floor = th[np.clip(cur, 0, 3)] + np.where(cur > 0, self.model.delta, 0) - pts
        cand = (~up) & (raw >= 0) & (np.nan_to_num(score, nan=-1) < floor) & (cur > 0)
        below = np.where(cand, np.where(np.isnan(below), now, below), np.nan)
        step = cand & (now - below >= mins)
        cur = np.where(step, cur - 1, cur)
        below = np.where(step, now, below)
        cur = np.where(raw < 0, -1, cur)
        self._cur, self._below = cur, below
        return cur.copy()

    def reset_state(self):
        self._cur[:] = -1
        self._below[:] = np.nan

    # ---------------- 설명·집계
    def detail(self, snap: Snapshot, bid: int) -> dict:
        r = self.model.compute(snap.ctx, idx=[bid])
        g = lambda k: [None if np.isnan(v) else round(float(v), 3) for v in r[k][:, 0]]
        return {"horizons": snap.ctx.horizons, "score": g("score"), "level": [int(v) for v in r["level"][:, 0]],
                "depth": g("depth"), "opening_excess": g("opening_excess"), "protected_excess": g("protected_excess"),
                "util": g("util"), "h_pred": g("h_pred"), "rain60": g("rain60"),
                "components": {k: g(f"s_{k}") for k in ("opening", "sewer", "rain", "terrain")},
                "t_flood": int(r["t_flood"][0]), "t_protected": int(r["t_protected"][0]),
                "confidence": int(r["confidence"][0]), "rise30": None if np.isnan(r["rise30"][0]) else float(r["rise30"][0])}

    def station_of(self, bid: int):
        j = int(self.st_idx[bid])
        return None if j < 0 else (j, self.stations[j], float(self.st_dist[bid]))

    def reasons(self, snap: Snapshot, bid: int) -> list[str]:
        d = self.detail(snap, bid)
        rf = snap.ctx.rain[0]
        out = []
        if rf is None:
            out.append("최근 60분 강우 자료가 부족함 → 강우 점수 제외 (0으로 간주하지 않음)")
        else:
            out.append(f"최근 60분 강우 {rf['P60']:.0f}mm, 30분 강도 {rf['I30']:.0f}mm/h")
        st = self.station_of(bid)
        if st is None:
            out.append(f"반경 {self.cfg['sewer']['station_radius_m']}m 안 하수관 수위계 없음 → 하수관 점수 제외")
        else:
            j, s, dist = st
            f = snap.ctx.station_feat[j]
            if f is None:
                out.append(f"수위계 {s['station']}({dist:.0f}m) 최근 자료 없음 → 하수관 점수 제외")
            else:
                rise = f"{f['rise30'] * 100:+.1f}cm/분" if f["rise30"] is not None else "상승속도 모름"
                out.append(f"하수관 {s['station']}({dist:.0f}m) 수위 {f['H0']:.2f}m ({rise}), "
                           f"관 용량 대비 {d['util'][0] * 100:.0f}% (기준 {self.h_crit[j]:.2f}m: {self.h_crit_src[j]})"
                           if d["util"][0] is not None else f"하수관 {s['station']} 수위 {f['H0']:.2f}m")
        if d["depth"][0] is not None:
            out.append(f"추정 지표면 물깊이 {d['depth'][0] * 100:.0f}cm, 개구부 초과 {d['opening_excess'][0] * 100:.0f}cm")
        rel = self.static.iloc[bid].rel_elev_m
        if not np.isnan(rel):
            out.append(f"주변 300m 평균보다 {abs(rel):.1f}m {'낮음' if rel < 0 else '높음'}")
        dg = self.cfg.get("district_gate") or {}
        if dg.get("enabled"):
            g = snap.ctx.gate
            rule = {"seoul2023": "침수예보: 1시간 55mm 또는 15분 20mm 초과",
                    "seoul2026": "침수경보: 1시간 50mm+3시간 90mm 또는 1시간 72mm"}
            if g is None:
                out.append("자치구 강우량계 자료 없음 → 자치구 기준 없이 건물 기준만으로 판정")
            else:
                out.append(f"자치구 강우 기준({rule.get(dg.get('rule'), dg.get('rule'))}) "
                           f"{'충족 → 건물 기준으로 경고 이상 가능' if g[0] else '미충족 → 최대 주의'}"
                           + (f", 60분 안 충족 예상" if (not g[0] and g.any()) else ""))
        if self.vulnerable[bid]:
            top = (self.cfg.get("vulnerable") or {}).get("top_pct", 10)
            out.append(f"평상시 취약 건물 (과거 침수·저지대·침수예상도 기준 신림동 반지하 상위 {top:g}%) — 비가 오면 먼저 확인할 곳")
        wm = self.cfg.get("watch_min")
        if wm and (rf is None or rf["P60"] < wm["rain60_mm"]) and int(snap.grades[bid]) <= 0:
            out.append(f"60분 강우 {wm['rain60_mm']:g}mm 미만·하수관 {wm['sewer_util'] * 100:.0f}% 미만이라 등급을 띄우지 않음 (비·물이 있을 때만 주의 이상)")
        dl = float(self.model.delta[bid])
        sh = float(dg.get("theta_shift", 0)) if dg.get("enabled") and snap.ctx.gate is not None else 0.0
        if dl or sh:
            out.append(f"건물 취약성(과거 침수·지형·구조·침수예상도) 반영 경고 기준 {self.cfg['levels'][1] + dl:.0f}점"
                       + (f", 구 발령 중 경고 판정 기준 {self.cfg['levels'][1] + dl + sh:.0f}점" if sh else ""))
        return out

    def counts(self, grades: np.ndarray) -> dict:
        g = grades[self.banjiha]
        return {LEVELS_KO[i]: int((g == i).sum()) for i in range(4)} | {"데이터 없음": int((g < 0).sum())}
