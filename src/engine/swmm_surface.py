"""SWMM 수식 기반 지표면 수심 (건물마다 가상 소유역 1개, 시간 적분).

경험식 SurfaceDepth = α·SewerExcess + β·max(0, I−I_drain) + γ·Lowland 를 SWMM 5의 계산식으로 대체한다.
SWMM 엔진(관망 동역학)을 돌리지 않고, 관망 상태는 하수관 수위계 실측(이용률)으로 대신한다.

1) 지표 유출 — 비선형 저류 (SWMM Runoff, Manning)
     dd/dt = i − f − q,   q = (W/A)·√S / n · max(0, d − ds)^(5/3)      [불투수·투수 소영역 따로]
     반암시적(선형화 후진 Euler)으로 풀어 큰 시간간격에도 안정
2) 침투 — Horton (투수 영역)
     f_cap = f_min + (f0 − f_min)·exp(−k·t_wet),   f = min(f_cap, 공급 가능량)
3) 우수받이·하수관 유입 용량 — 합리식 설계용량 × 하수관 여유
     Q_cap = i_design·A · clip((u_full − U)/(u_full − u_start), 0, 1)       U = 하수관 이용률(수위/만관수위)
4) 맨홀 역류 — SWMM 오리피스식
     Q_back = Cd·A_o·√(2g·h_s),   h_s = max(0, U − 1)·H_crit    (만관 이후 압력수두)
5) 저지대 유입 — SWMM 소유역 연결(RouteTo): 윗동네에서 넘친 물이 낮은 곳으로
     Q_up = up_ratio·Lowland·max(0, Q_runoff − Q_cap)
6) 지표 저류 — SWMM 맨홀 담수(Ponded area)
     V ← max(0, V + Δt·(Q_runoff + Q_up + Q_back − Q_drain − Q_street)),  Q_drain = min(Q_cap, 유입 + V/Δt)
     SurfaceDepth = V / A_pond,   A_pond = pond_frac·A
7) 도로 흘러내림 — 고인 물이 도로 경사를 따라 빠짐 (Manning 광폭수로, 저지대일수록 갇힘)
     Q_street = (1 − Lowland)·B/n_s·√S·depth^(5/3)

기본 계수는 data/replay/sillim419_pilot.inp(신림동 419 시범 SWMM 모델) 값과 같고, 모두 settings.yaml surface.swmm
(보정 대상: i_design_mmh, pond_frac, up_ratio, orifice_cda_m2, u_start).
"""
import numpy as np

G = 9.81


def _reservoir(d, i, coef, ds, dt):
    """비선형 저류 한 단계(반암시적). d·i 단위 m, m/s. 반환 (새 d, 평균 유출률 m/s)."""
    x = np.maximum(d - ds, 0.0)
    q = coef * x ** (5 / 3)
    dq = coef * (5 / 3) * x ** (2 / 3)
    d_new = np.maximum(0.0, d + dt * (i - q) / (1 + dt * dq))
    # 질량보존: 유출 = 유입 − 저류변화
    out = np.maximum(0.0, i - (d_new - d) / dt)
    return d_new, out


class SurfaceState:
    """건물별 상태(불투수·투수 수심, 습윤시간, 지표 저류량). 시뮬레이션마다 새로 만든다.
    exact 모드: d_imp0 = 저류 없는 불투수(A3) 수심, tp = Horton 등가 침투시간(s)."""

    def __init__(self, n):
        self.d_imp = np.zeros(n)
        self.d_imp0 = np.zeros(n)
        self.d_perv = np.zeros(n)
        self.t_wet = np.zeros(n)
        self.tp = np.zeros(n)
        self.V = np.zeros(n)


def horton_exact(tp, ia, dt, f0, fmin, kd, kr):
    """SWMM 5 적분형 Horton (EPA/600/R-15/162A §4.2.3 계산 절차 p.94, 식 4-3·4-5·4-11).
    tp 등가 침투시간(s), ia = i + d/Δt 공급 가능률(m/s). 반환 (새 tp, 침투율 f m/s). 배열 연산."""
    df = f0 - fmin
    F = lambda t: fmin * t + df / kd * (1 - np.exp(-kd * t))                          # 식 4-3
    tlim = 16.0 / kd
    dry = ia <= 0
    # 1) 공급 없음 → 회복: tp ← −ln(1 − e^{−kr·Δt}(1 − e^{−kd·tp}))/kd  (식 4-11을 tp로 푼 형태)
    tp_rec = -np.log(np.maximum(1e-12, 1 - np.exp(-kr * dt) * (1 - np.exp(-kd * tp)))) / kd
    # 2) 공급 있음 → Δt 동안 가능 침투량 Fp→F1
    flat = tp >= tlim
    Fp = np.where(flat, fmin * tp + df / kd, F(tp))
    F1 = np.where(flat, Fp + fmin * dt, F(tp + dt))
    fp = np.maximum((F1 - Fp) / dt, fmin)
    limited = (tp + dt > tlim) | (fp < ia)          # 절차 5: t1 > 16/kd 또는 fp < ia → tp += Δt
    # 공급이 작으면 F(t1) = Fp + ia·Δt 를 Newton–Raphson으로 풀어 tp = t1 (식 4-5)
    target = Fp + ia * dt
    t1 = tp + dt
    for _ in range(20):
        g = F(t1) - target
        dg = fmin + df * np.exp(-kd * t1)
        t1 = np.clip(t1 - g / dg, tp, tp + dt)
    tp_new = np.where(dry, tp_rec, np.where(limited, tp + dt, t1))
    f = np.where(dry, 0.0, np.minimum(fp, ia))                                         # 절차 7
    return tp_new, f


def params(cfg: dict, slope_pct, lowland):
    s = cfg["swmm"]
    A, W = s["area_m2"], s["width_m"]
    S = np.clip(np.nan_to_num(np.asarray(slope_pct, dtype=float), nan=s["slope_default_pct"]),
                s["slope_min_pct"], s["slope_max_pct"]) / 100
    exact = bool(s.get("exact", False))
    imp = s["imperv"]
    # exact: SWMM 식 3-8·3-9 — α는 소영역 면적으로 나눈다 (A_I = imperv·A, A_P = (1−imperv)·A)
    a_imp, a_perv = (imp * A, (1 - imp) * A) if exact else (A, A)
    return {"A": A, "imperv": imp, "exact": exact, "pct_zero": s.get("pct_zero", 0.25) if exact else 0.0,
            "c_imp": (W / a_imp) * np.sqrt(S) / s["n_imperv"], "c_perv": (W / a_perv) * np.sqrt(S) / s["n_perv"],
            "ds_imp": s["ds_imperv_mm"] / 1000, "ds_perv": s["ds_perv_mm"] / 1000,
            "kr": 3.912 / (s.get("dry_days", 7) * 86400),                               # 식 4-14
            "f0": s["horton_f0_mmh"] / 3.6e6, "fmin": s["horton_fmin_mmh"] / 3.6e6, "k": s["horton_decay_h"] / 3600,
            "q_design": s["i_design_mmh"] / 3.6e6 * A, "u_start": s["u_start"], "u_full": s["u_full"],
            "cda": s["orifice_cda_m2"], "up": s["up_ratio"] * np.asarray(lowland, dtype=float),
            "A_pond": s["pond_frac"] * A, "sub_s": s["substep_s"],
            "c_street": (1 - np.asarray(lowland, dtype=float)) * s["street_width_m"] / s["n_street"] * np.sqrt(S)}


def step(st: SurfaceState, p: dict, i_ms, util, h_crit, dt):
    """Δt(초) 동안 강우강도 i_ms(m/s), 하수관 이용률 util(NaN = 자료 없음 → 하수관 정상 가정)."""
    if p["exact"]:
        # SWMM 3개 소영역 (매뉴얼 §3.2 그림 3-3): A1 투수, A2 불투수(저류 ds), A3 불투수(저류 0, PctZero)
        st.d_imp, q_imp2 = _reservoir(st.d_imp, i_ms, p["c_imp"], p["ds_imp"], dt)
        st.d_imp0, q_imp3 = _reservoir(st.d_imp0, i_ms, p["c_imp"], 0.0, dt)
        q_imp = (1 - p["pct_zero"]) * q_imp2 + p["pct_zero"] * q_imp3
        ia = i_ms + st.d_perv / dt
        st.tp, f = horton_exact(st.tp, ia, dt, p["f0"], p["fmin"], p["k"], p["kr"])
        st.d_perv, q_perv = _reservoir(st.d_perv, i_ms - f, p["c_perv"], p["ds_perv"], dt)   # ix = i − f (음수 허용)
    else:
        st.d_imp, q_imp = _reservoir(st.d_imp, i_ms, p["c_imp"], p["ds_imp"], dt)
        # Horton: 공급(강우+저류수)이 있을 때만 습윤시간 증가
        f_cap = p["fmin"] + (p["f0"] - p["fmin"]) * np.exp(-p["k"] * st.t_wet)
        supply = i_ms + st.d_perv / dt
        f = np.minimum(f_cap, supply)
        st.t_wet = st.t_wet + np.where(supply > 0, dt, 0.0)
        st.d_perv, q_perv = _reservoir(st.d_perv, np.maximum(0.0, i_ms - f), p["c_perv"], p["ds_perv"], dt)
    q_run = p["A"] * (p["imperv"] * q_imp + (1 - p["imperv"]) * q_perv)          # m³/s

    u = np.asarray(util, dtype=float)
    known = ~np.isnan(u)
    free = np.where(known, np.clip((p["u_full"] - u) / (p["u_full"] - p["u_start"]), 0.0, 1.0), 1.0)
    q_cap = p["q_design"] * free
    h_s = np.where(known, np.maximum(0.0, u - 1.0) * np.nan_to_num(h_crit), 0.0)
    q_back = p["cda"] * np.sqrt(2 * G * h_s)
    q_up = p["up"] * np.maximum(0.0, q_run - p["q_design"] * free)

    inflow = q_run + q_up + q_back
    q_drain = np.where(h_s > 0, 0.0, np.minimum(q_cap, inflow + st.V / dt))     # 역류 중엔 받이로 못 빠짐
    V = np.maximum(0.0, st.V + dt * (inflow - q_drain))
    # 도로 흘러내림: 반암시적 (저류식과 같은 방식) — V = A_pond·y
    y = V / p["A_pond"]
    c = p["c_street"] / p["A_pond"]
    q = c * y ** (5 / 3)
    dq = c * (5 / 3) * y ** (2 / 3)
    y = np.maximum(0.0, y - dt * q / (1 + dt * dq))
    st.V = y * p["A_pond"]
    return y


def simulate(cfg: dict, rain_mm, dt_min, util, h_crit, slope_pct, lowland, state: SurfaceState | None = None):
    """rain_mm [n,S] 또는 [S]: 각 단계 끝 10분 강수(mm). dt_min [S]: 앞 단계와의 간격(분, 첫 단계 10).
    util [n,S] / 스칼라 NaN, h_crit [n] → 각 단계 끝 지표면 수심 [n,S] (m)."""
    rain_mm = np.asarray(rain_mm, dtype=float)
    S = rain_mm.shape[-1]
    n = len(np.atleast_1d(slope_pct))
    p = params(cfg, slope_pct, lowland)
    st = state or SurfaceState(n)
    util = np.broadcast_to(np.asarray(util, dtype=float), (n, S)) if np.ndim(util) else np.full((n, S), np.nan)
    rain = np.broadcast_to(rain_mm, (n, S)) if rain_mm.ndim == 1 else rain_mm
    out = np.zeros((n, S))
    sub = max(1, int(round(600 / p["sub_s"])))
    for k in range(S):
        gap = float(dt_min[k]) - 10
        if gap > 0:                                           # 비 없는 구간은 한 번에 배수
            step(st, p, 0.0, util[:, k], h_crit, gap * 60)
        i = np.nan_to_num(rain[:, k]) / 1000 / 600
        for _ in range(sub):
            depth = step(st, p, i, util[:, k], h_crit, 600 / sub)
        out[:, k] = depth
    return out
