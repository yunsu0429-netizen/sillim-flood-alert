"""반지하 침수위험 판정 모델 v0.1 (basement_flood_risk_logic_prototype.md §1–23).

강우 관측/예보 → 강우 특성 → 하수관 수위 예측 → 지표면 물높이 → 지반고·개구부 비교 → 4요소 점수 → 등급·override
→ 신뢰도 → 10~60분 예보·침수 예상시점.

원칙
- 하수관 수위와 지표면 침수수위를 같게 보지 않는다 (H_sewer ≠ W_surface).
- 개구부 높이는 바깥 지면 기준 높이(문턱)로 저장되므로 W_surface − Z_opening = SurfaceDepth − 문턱높이 (GL 오차 상쇄).
- 결측은 0으로 채우지 않는다: 점수 요소가 없으면 남은 가중치로 재정규화하고 신뢰도를 낮춘다.
- 모든 계수·임계값은 config/settings.yaml risk_model (보정 전 가정값).
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

LEVELS = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
LEVELS_KO = ["낮음", "주의", "경고", "위험"]


# ---------------------------------------------------------------- 점수표 (스칼라·배열 공용)
def opening_score(depth):
    """§14 A: depth<=0 → 0, 0~0.10 → 20, 0.10~0.30 → 50, 0.30~0.50 → 75, >0.50 → 100."""
    d = np.asarray(depth, dtype=float)
    out = np.select([d <= 0, d <= 0.10, d <= 0.30, d <= 0.50], [0, 20, 50, 75], 100).astype(float)
    return np.where(np.isnan(d), np.nan, out)


def _table_below(x, table):
    """table = [[상한, 점수], ...] 오름차순. x < 상한인 첫 구간의 점수."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, float(table[-1][1]))
    for ub, sc in reversed(table):
        out = np.where(x < ub, float(sc), out)
    return np.where(np.isnan(x), np.nan, out)


def sewer_score(util, table):
    return _table_below(util, table)


def rain_score(rain60_mm, table):
    return _table_below(rain60_mm, table)


def terrain_score(rel_elev_m, table):
    return _table_below(rel_elev_m, table)


def weighted_score(components: dict, weights: dict):
    """§15 가중합. 값이 NaN인 요소는 빼고 남은 가중치 합으로 나눈다 (재정규화)."""
    num = 0.0
    den = 0.0
    for k, v in components.items():
        v = np.asarray(v, dtype=float)
        ok = ~np.isnan(v)
        num = num + np.where(ok, weights[k] * v, 0.0)
        den = den + np.where(ok, weights[k], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def classify(score, opening_excess, levels, ov, rise30=None, rain60=None, offset=None):
    """§16 등급 + §17 override. score NaN → -1 (판정 불가).
    offset: 건물별 경계 조정 Δ(D. 정적 취약성, docs/references.md §7) — 경계 20/40/70에 모두 더한다."""
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=0)
    off = 0.0 if offset is None else np.asarray(offset, dtype=float)
    lv = sum((s >= L + off).astype(int) for L in levels)
    s = np.asarray(score, dtype=float)
    ex = np.asarray(opening_excess, dtype=float)
    lv = np.where(ex > 0, np.maximum(lv, 2), lv)                         # 개구부 초과 → 최소 HIGH
    lv = np.where(ex >= ov["opening_very_high_m"], 3, lv)                # 0.30m 이상 → VERY_HIGH
    if rise30 is not None and rain60 is not None:
        r = np.asarray(rise30, dtype=float)
        crit = (np.nan_to_num(r, nan=-1) >= ov["critical_rise_m_per_min"]) & (np.nan_to_num(rain60, nan=-1) >= ov["heavy_rain60_mm"])
        lv = np.where(crit, np.maximum(lv, 2), lv)
    return np.where(np.isnan(s), -1, lv).astype(int)


def official_rule(p10: np.ndarray, rule: str) -> np.ndarray:
    """서울시 침수예보 강우 기준(자치구 발령)을 10분 강우 [강우계, 시각]에 적용 → 시각별 충족 여부 [강우계, 시각].
    seoul2023: 침수예보 — 1시간 55mm 초과 또는 15분 20mm 초과 (2023-05~, 서울시 2023-03-10 보도
               news.seoul.go.kr/env/archives/522983). 15분은 10분 칸 안 균등 가정 max(P[t]+½P[t+1], ½P[t−1]+P[t])
    seoul2026: 침수경보 — 1시간 50mm 이상 + 3시간 90mm 이상, 또는 1시간 72mm 이상(2026 강화, 즉시 재난문자)
               (아시아경제 2026-07-18, 헤럴드경제 2026-05-10 보도; docs/references.md §10)"""
    p = np.nan_to_num(np.atleast_2d(np.asarray(p10, dtype=float)))
    c = np.cumsum(np.pad(p, ((0, 0), (1, 0))), axis=1)
    win = lambda k: c[:, 1:] - c[:, np.maximum(np.arange(p.shape[1]) + 1 - k, 0)]
    p60, p180 = win(6), win(18)
    if rule == "seoul2023":
        nxt = np.pad(p[:, 1:], ((0, 0), (0, 1)))
        prv = np.pad(p[:, :-1], ((0, 0), (1, 0)))
        s15 = np.maximum(p + 0.5 * nxt, 0.5 * prv + p)
        return (p60 > 55) | (s15 > 20)
    if rule == "seoul2026":
        return (p60 >= 72) | ((p60 >= 50) & (p180 >= 90))
    raise ValueError(rule)


def gate_levels(lv_gated, lv_plain, gate):
    """§10 AND 결합. '주의'는 원래 경계(lv_plain) 그대로, '경고' 이상은 구 발령(gate) 중일 때만 이동된 경계(lv_gated)로.
    구 발령이 없으면 최대 '주의'. 구 발령이 켜져도 '주의'가 '낮음'으로 내려가지 않는다."""
    watch = np.where(lv_plain < 0, lv_plain, np.minimum(lv_plain, 1))
    return np.where(gate, np.where(lv_gated >= 2, lv_gated, watch), watch)


def surface_depth(sewer_excess, rain_intensity, lowland, cfg):
    """§9 SurfaceDepth (보정 두 가지는 설정으로 끌 수 있음 — settings.yaml surface 참고)."""
    se = np.nan_to_num(np.asarray(sewer_excess, dtype=float), nan=0.0)
    rain_ex = np.maximum(0.0, np.asarray(rain_intensity, dtype=float) - cfg["rain_drain_capacity_mmh"])
    water = cfg["alpha"] * se + cfg["beta"] * rain_ex
    low = cfg["gamma"] * np.asarray(lowland, dtype=float)
    if cfg.get("lowland_needs_water", True):
        low = np.where(water > 0, low, 0.0)
    return np.maximum(0.0, water + low)


def excess_over(depth, sill):
    """§10–12 max(0, W − Z) 를 지면 기준으로: max(0, depth − 문턱높이)."""
    return np.maximum(0.0, np.asarray(depth, dtype=float) - np.asarray(sill, dtype=float))


def confidence_level(score, levels):
    return np.where(np.asarray(score) >= levels[1], "HIGH", np.where(np.asarray(score) >= levels[0], "MEDIUM", "LOW"))


# ---------------------------------------------------------------- 강우·하수관 특성
def rain_window(series, end: datetime, minutes: int):
    """(end−minutes, end] 10분 강우 합. series가 그 구간을 하나도 못 덮으면 None."""
    vals = [v for ts, v in series if end - timedelta(minutes=minutes) < ts <= end]
    return sum(vals) if vals else None


def rain_features(series, t: datetime) -> dict | None:
    """§5 P10·P30·P60·P180, I10·I30·I60(mm/h)."""
    p = {m: rain_window(series, t, m) for m in (10, 30, 60, 180)}
    if p[60] is None:
        return None
    p10 = p[10] if p[10] is not None else 0.0
    p30 = p[30] if p[30] is not None else 0.0
    return {"P10": p10, "P30": p30, "P60": p[60], "P180": p[180] or 0.0,
            "I10": p10 * 6, "I30": p30 * 2, "I60": p[60]}


def _level_near(obs, when: datetime, tol_min: float):
    best = None
    for ts, lv, _ in obs:
        d = abs((ts - when).total_seconds()) / 60
        if d <= tol_min and (best is None or d < best[0]):
            best = (d, lv)
    return None if best is None else best[1]


def sewer_features(obs, t: datetime, stale_min: float) -> dict | None:
    """§6 H0·H10·H30, RiseRate10·30(m/min), 최근 30분 선형추세. 최신값이 stale_min보다 오래되면 None."""
    past = [o for o in obs if o[0] <= t]
    if not past or (t - past[-1][0]).total_seconds() / 60 > stale_min:
        return None
    t0, h0, status = past[-1]
    h10 = _level_near(past, t0 - timedelta(minutes=10), 5)
    h30 = _level_near(past, t0 - timedelta(minutes=30), 7)
    r10 = (h0 - h10) / 10 if h10 is not None else None
    r30 = (h0 - h30) / 30 if h30 is not None else r10
    win = [(ts, lv) for ts, lv, _ in past if ts >= t0 - timedelta(minutes=30)]
    trend = None
    if len(win) >= 3:
        x = np.array([(ts - t0).total_seconds() / 60 for ts, _ in win])
        trend = float(np.polyfit(x, [lv for _, lv in win], 1)[0])
    return {"H0": h0, "H10": h10, "H30": h30, "rise10": r10, "rise30": r30, "trend": trend,
            "last_ts": t0, "status": status}


def predict_sewer_arx(coef, H0: float, H10: float, rain_past, rain_future, steps: int, h_crit: float,
                      cap_ratio: float) -> list:
    """C. ARX 강우-수위 반응 모형으로 10분씩 굴려 steps칸 예측 (docs/references.md §6).
    H(t+1) = a·H(t) + b·H(t−1) + Σ_{k=0..5} c_k·r(t−k) + d,  rain_past = 최근 6칸(오래된→최근), rain_future = 앞으로.
    매 주기 실측 H0·H10에서 다시 시작(관측 오차가 작은 칼만 필터와 같음). 0 ~ cap·H_crit로 제한."""
    lags = len(coef) - 3
    rp = list(rain_past)[-lags:]
    rp = [0.0] * (lags - len(rp)) + rp
    h_prev, h, out = H10, H0, []
    for k in range(steps):
        lag = rp[-lags:][::-1]
        nxt = coef[0] * h + coef[1] * h_prev + sum(c * r for c, r in zip(coef[2:2 + lags], lag)) + coef[-1]
        nxt = float(np.clip(nxt, 0.0, cap_ratio * h_crit))
        out.append(nxt)
        h_prev, h = h, nxt
        rp.append(float(rain_future[k]) if k < len(rain_future) else 0.0)
    return out


def predict_sewer(f: dict, h: int, h_crit: float, cap_ratio: float) -> float:
    """§7 H_pred = H0 + RiseRate30·h, 0 ~ cap·H_crit 사이로 제한. 상승속도를 모르면 현재값 유지."""
    rise = f["rise30"] if f["rise30"] is not None else 0.0
    return float(np.clip(f["H0"] + rise * h, 0.0, cap_ratio * h_crit))


# ---------------------------------------------------------------- 한 시점의 계산 맥락
@dataclass
class Context:
    """한 판정 시각 t의 모든 시점(h) 공통 입력. 건물별 값은 compute()로 만든다."""
    t: datetime
    horizons: list
    rain: dict                       # h → rain_features 또는 None
    rain_quality: float              # 0~100
    station_ids: list
    station_feat: list               # 수위계별 sewer_features 또는 None
    h_crit: np.ndarray               # 수위계별 H_critical
    h_crit_src: list                 # 'override' | '2022-08-08 최고수위' | '중앙값 대체'
    h_pred: np.ndarray               # [시점, 수위계]
    util: np.ndarray                 # [시점, 수위계]
    meta: dict = field(default_factory=dict)
    # SWMM 지표면 적분용 10분 시계열 (surface.method == swmm): rain_mm [S], dt_min [S], util [S, 수위계],
    # h_index = 각 예보 시점(h)의 단계 번호
    surface_series: dict | None = None
    # 자치구 침수예보 기준 충족 여부 [시점] (district_gate, 서울시 강우량계). None = 자료 없음 → 게이트 미적용
    gate: np.ndarray | None = None


class RiskModel:
    def __init__(self, cfg: dict, static, station_index: np.ndarray, station_dist: np.ndarray,
                 dem_quality: float):
        self.cfg = cfg
        self.n = len(static)
        self.banjiha = static["is_banjiha"].to_numpy()
        self.rel = static["rel_elev_m"].to_numpy(dtype=float)
        self.lowland = np.clip(-self.rel / cfg["surface"]["lowland_full_m"], 0, 1)
        self.lowland = np.nan_to_num(self.lowland, nan=0.0)
        self.delta = np.zeros(self.n)              # D. 건물별 경계 조정 (RiskEngine이 static_prior로 채움)
        self.slope = static["slope_pct"].to_numpy(dtype=float) if "slope_pct" in static else np.full(self.n, np.nan)
        self.st_idx = station_index           # 건물별 수위계 번호 (-1 = 반경 안에 없음)
        self.st_dist = station_dist
        self.dem_quality = dem_quality
        self.set_openings({}, None)

    # ---- 개구부
    def set_openings(self, per_bid: dict, imputed: dict | None):
        """per_bid: bid → {sill, barrier, quality, source}. 없는 건물은 imputed(평균 대체값)."""
        self.sill = np.full(self.n, np.nan)
        self.barrier = np.zeros(self.n)
        self.op_quality = np.array(["none"] * self.n, dtype=object)
        if imputed:
            self.sill[:] = imputed["sill"]
            self.barrier[:] = 0.0
            self.op_quality[:] = "imputed"
        for bid, o in per_bid.items():
            self.sill[bid] = o["sill"]
            self.barrier[bid] = o.get("barrier") or 0.0
            self.op_quality[bid] = o["quality"]
        self.imputed = imputed

    # ---- 계산
    def compute(self, ctx: Context, idx=None) -> dict:
        """건물 배열(idx 지정 시 그 건물들만) × 시점별 결과."""
        c = self.cfg
        sel = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        st = self.st_idx[sel]
        has_st = st >= 0
        H = len(ctx.horizons)
        shape = (H, len(sel))
        out = {k: np.full(shape, np.nan) for k in
               ("score", "depth", "opening_excess", "protected_excess", "util", "h_pred", "rain60",
                "s_opening", "s_sewer", "s_rain", "s_terrain")}
        level = np.full(shape, -1, dtype=int)
        rise30 = np.array([ctx.station_feat[s]["rise30"] if s >= 0 and ctx.station_feat[s] and
                           ctx.station_feat[s]["rise30"] is not None else np.nan for s in st])
        t_terrain = terrain_score(self.rel[sel], c["scores"]["terrain"])
        swmm_depth = self._swmm_depth(ctx, sel, st, has_st)
        for k, h in enumerate(ctx.horizons):
            rf = ctx.rain[h]
            util = np.where(has_st, ctx.util[k][np.maximum(st, 0)], np.nan) if len(ctx.station_ids) else np.full(len(sel), np.nan)
            hp = np.where(has_st, ctx.h_pred[k][np.maximum(st, 0)], np.nan) if len(ctx.station_ids) else np.full(len(sel), np.nan)
            hc = np.where(has_st, ctx.h_crit[np.maximum(st, 0)], np.nan) if len(ctx.station_ids) else np.full(len(sel), np.nan)
            sewer_ex = np.maximum(0.0, hp - hc)
            if rf is None and np.all(np.isnan(util)):
                continue                                             # 강우도 하수관도 없으면 판정 불가
            intensity = rf["I30"] if rf else 0.0
            if swmm_depth is not None:
                depth = swmm_depth[:, ctx.surface_series["h_index"][k]]
            else:
                depth = surface_depth(sewer_ex, intensity, self.lowland[sel], c["surface"])
            ex = excess_over(depth, self.sill[sel])
            pex = excess_over(depth, self.sill[sel] + self.barrier[sel])
            r60 = rf["P60"] if rf else np.nan
            comps = {"opening": opening_score(ex),
                     "sewer": sewer_score(util, c["scores"]["sewer"]),
                     "rain": np.full(len(sel), rain_score(r60, c["scores"]["rain60"]) if rf else np.nan),
                     "terrain": t_terrain}
            score = weighted_score(comps, c["weights"])
            score = np.where(self.banjiha[sel], score, np.nan)
            lv = classify(score, np.where(self.banjiha[sel], ex, np.nan), c["levels"], c["override"],
                          rise30, np.full(len(sel), r60), offset=self.delta[sel])
            dg = c.get("district_gate") or {}
            if dg.get("enabled") and ctx.gate is not None:
                lv_g = classify(score, np.where(self.banjiha[sel], ex, np.nan), c["levels"], c["override"],
                                rise30, np.full(len(sel), r60), offset=self.delta[sel] + dg.get("theta_shift", 0))
                lv = gate_levels(lv_g, lv, bool(ctx.gate[k]))
            out["score"][k], out["depth"][k], out["opening_excess"][k] = score, depth, ex
            out["protected_excess"][k], out["util"][k], out["h_pred"][k] = pex, util, hp
            out["rain60"][k] = r60
            out["s_opening"][k], out["s_sewer"][k] = comps["opening"], comps["sewer"]
            out["s_rain"][k], out["s_terrain"][k] = comps["rain"], comps["terrain"]
            level[k] = lv
        out["level"] = level
        out["t_flood"] = self._first(out["opening_excess"], ctx.horizons)
        out["t_protected"] = self._first(out["protected_excess"], ctx.horizons)
        out["confidence"] = self._confidence(ctx, sel)
        out["rise30"] = rise30
        return out

    def _swmm_depth(self, ctx: Context, sel, st, has_st):
        """SWMM 수식으로 판정 시각 전 warmup부터 +60분까지 적분 → [건물, 단계] 지표면 수심. 해당 없으면 None."""
        ss = ctx.surface_series
        if self.cfg["surface"].get("method") != "swmm" or not ss:
            return None
        from src.engine import swmm_surface
        S = len(ss["rain_mm"])
        if len(ctx.station_ids):
            util = np.where(has_st[:, None], np.asarray(ss["util"]).T[np.maximum(st, 0)], np.nan)
            hc = np.where(has_st, ctx.h_crit[np.maximum(st, 0)], np.nan)
        else:
            util, hc = np.full((len(sel), S), np.nan), np.full(len(sel), np.nan)
        out = np.zeros((len(sel), S))                    # 반지하 후보만 적분 (나머지는 점수 대상이 아님)
        m = self.banjiha[sel]
        if m.any():
            out[m] = swmm_surface.simulate(self.cfg["surface"], ss["rain_mm"], ss["dt_min"], util[m], hc[m],
                                           self.slope[sel][m], self.lowland[sel][m])
        return out

    @staticmethod
    def _first(excess, horizons):
        """§13 T_flood = 처음 excess > 0 이 되는 시점(분). 없으면 -1."""
        hit = np.nan_to_num(excess, nan=0) > 0
        first = np.argmax(hit, axis=0)
        return np.where(hit.any(axis=0), np.array(horizons)[first], -1)

    def _confidence(self, ctx: Context, sel):
        cc = self.cfg["confidence"]
        sq = cc["sewer_quality"]
        st = self.st_idx[sel]
        sewer_q = np.array([
            (sq["missing"] if s < 0 or ctx.station_feat[s] is None else
             sq["observed"] if ctx.h_crit_src[s] in ("override", "2022-08-08 최고수위") else sq["observed_hcrit_estimated"])
            for s in st], dtype=float)
        oq = cc["opening_quality"]
        build_q = np.array([oq.get(q, 0) for q in self.op_quality[sel]], dtype=float)
        w = cc["weights"]
        score = (w["sewer"] * sewer_q + w["rain"] * ctx.rain_quality + w["elevation"] * self.dem_quality
                 + w["building"] * build_q)
        return np.round(score).astype(int)
