"""SWMM 수식 지표면 모듈 (src/engine/swmm_surface.py)."""
import numpy as np

from src.config import load_params
from src.engine import swmm_surface

CFG = load_params()["risk_model"]["surface"]


def run(rain_mmh, steps=12, util=np.nan, h_crit=1.0, slope=5.0, lowland=0.0):
    rain = np.full(steps, rain_mmh / 6)          # 10분 강수(mm)
    u = np.full((1, steps), util)
    return swmm_surface.simulate(CFG, rain, np.full(steps, 10.0), u, np.array([h_crit]), np.array([slope]),
                                 np.array([lowland]))[0]


def test_no_rain_no_water():
    assert run(0).max() == 0


def test_rain_below_design_capacity_drains():
    assert run(CFG["swmm"]["i_design_mmh"] * 0.5).max() < 0.005


def test_rain_above_design_ponds_and_lowland_traps_more():
    hi = CFG["swmm"]["i_design_mmh"] * 2
    flat_low = run(hi, lowland=1.0)
    slope_high = run(hi, lowland=0.0)
    assert flat_low.max() > 0.05
    assert flat_low.max() > slope_high.max()


def test_full_sewer_blocks_drainage_and_backflows():
    r = CFG["swmm"]["i_design_mmh"] * 0.5
    free = run(r, util=0.3)
    full = run(r, util=1.3)
    assert full.max() > free.max()


def test_mass_never_negative_and_recedes_after_rain():
    hi = CFG["swmm"]["i_design_mmh"] * 2
    rain = np.r_[np.full(6, hi / 6), np.zeros(18)]
    d = swmm_surface.simulate(CFG, rain, np.full(24, 10.0), np.full((1, 24), np.nan), np.array([1.0]),
                              np.array([5.0]), np.array([0.5]))[0]
    assert (d >= 0).all()
    assert d[-1] < d.max()


def test_gap_step_drains_like_dry_period():
    hi = CFG["swmm"]["i_design_mmh"] * 2
    rain = np.r_[np.full(6, hi / 6), 0.0]
    a = swmm_surface.simulate(CFG, rain, np.r_[np.full(6, 10.0), 180.0], np.nan, np.array([np.nan]),
                              np.array([5.0]), np.array([0.5]))[0]
    assert a[-1] < a[-2]


# ---- exact 모드: SWMM 5 매뉴얼(EPA/600/R-15/162A) 원식과 대조
F0, FMIN, KD = 75 / 3.6e6, 10 / 3.6e6, 4 / 3600
KR = 3.912 / (7 * 86400)


def F_closed(t):                                   # 식 4-3
    return FMIN * t + (F0 - FMIN) / KD * (1 - np.exp(-KD * t))


def test_horton_exact_saturating_rain_follows_eq_4_3():
    tp, tot = np.zeros(1), 0.0
    for _ in range(36):                            # 큰 비 3시간, 5분 간격
        tp, f = swmm_surface.horton_exact(tp, np.array([200 / 3.6e6]), 300, F0, FMIN, KD, KR)
        tot += f[0] * 300
    assert abs(tot - F_closed(3 * 3600)) / F_closed(3 * 3600) < 1e-6


def test_horton_exact_light_rain_all_infiltrates_and_tp_matches_volume():
    tp = np.zeros(1)
    ia = 5 / 3.6e6                                 # 공급 < f∞ → 전량 침투, F(tp) = 누적 침투량 (식 4-5)
    for _ in range(12):
        tp, f = swmm_surface.horton_exact(tp, np.array([ia]), 300, F0, FMIN, KD, KR)
        assert np.isclose(f[0], ia)
    assert np.isclose(F_closed(tp[0]), ia * 3600, rtol=1e-6)


def test_horton_exact_recovers_when_dry():
    tp = np.array([3600.0])
    tp1, f = swmm_surface.horton_exact(tp, np.zeros(1), 86400, F0, FMIN, KD, KR)
    assert f[0] == 0 and 0 < tp1[0] < tp[0]
    tp2, _ = swmm_surface.horton_exact(tp, np.zeros(1), 30 * 86400, F0, FMIN, KD, KR)
    assert tp2[0] < 1.0                            # 오래 마르면 초기 용량으로


def test_exact_alpha_uses_subarea_area():
    import copy
    c = copy.deepcopy(CFG); c["swmm"]["exact"] = True
    p0, p1 = swmm_surface.params(CFG, 5.0, 0.0), swmm_surface.params(c, 5.0, 0.0)
    imp = CFG["swmm"]["imperv"]
    assert np.isclose(p1["c_imp"], p0["c_imp"] / imp)          # 식 3-9: W/(A2+A3)
    assert np.isclose(p1["c_perv"], p0["c_perv"] / (1 - imp))  # 식 3-8: W/A1
