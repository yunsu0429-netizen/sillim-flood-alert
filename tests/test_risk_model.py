"""basement_flood_risk_logic_prototype.md §23 테스트 + override·예측·알림 규칙."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.alerts.detector import evaluate
from src.config import load_params
from src.engine.risk_model import (classify, excess_over, opening_score, predict_sewer, rain_score, sewer_score,
                                   surface_depth, terrain_score, weighted_score)

P = load_params()
C = P["risk_model"]
T0 = datetime(2022, 8, 8, 20, 0)


# ---- §23 Test 1~5 (W·GL은 절대표고, 개구부는 GL 기준 문턱으로 바꿔 같은 식으로 계산)
def test1_water_below_ground_no_depth():
    assert excess_over(38.4 - 38.5, 0) == 0          # W=38.4, GL=38.5 → depth 0


def test2_ground_depth():
    assert excess_over(39.0 - 38.5, 0) == pytest.approx(0.5)


def test3_below_opening():
    assert excess_over(38.6, 38.7) == 0               # W=38.6, opening=38.7


def test4_opening_excess_forces_high():
    ex = excess_over(38.8, 38.7)
    assert ex == pytest.approx(0.1)
    assert classify(np.array([5.0]), np.array([ex]), C["levels"], C["override"])[0] >= 2


def test5_barrier_protects():
    assert excess_over(38.9, 38.7 + 0.3) == 0         # opening 38.7 + barrier 0.3, water 38.9


def test6_missing_component_is_not_zero():
    # 하수관 자료가 없으면 0점으로 넣지 않고 남은 가중치로 재정규화
    s = weighted_score({"opening": np.array([100.0]), "sewer": np.array([np.nan]),
                        "rain": np.array([100.0]), "terrain": np.array([100.0])}, C["weights"])
    assert s[0] == pytest.approx(100)
    s0 = weighted_score({"opening": np.array([0.0]), "sewer": np.array([np.nan]),
                         "rain": np.array([np.nan]), "terrain": np.array([np.nan])}, C["weights"])
    assert s0[0] == 0
    allnan = weighted_score({"opening": np.array([np.nan])}, C["weights"])
    assert np.isnan(allnan[0]) and classify(allnan, np.array([np.nan]), C["levels"], C["override"])[0] == -1


# ---- 점수표 경계
def test_score_tables():
    assert list(opening_score([0, 0.05, 0.10, 0.2, 0.3, 0.5, 0.6])) == [0, 20, 20, 50, 50, 75, 100]
    assert list(sewer_score([0.5, 0.7, 0.95, 1.0], C["scores"]["sewer"])) == [0, 30, 60, 100]
    assert list(rain_score([5, 10, 45, 79, 80], C["scores"]["rain60"])) == [0, 25, 50, 75, 100]
    assert list(terrain_score([1.0, 0.7, 0.2, -0.5, -0.6], C["scores"]["terrain"])) == [0, 20, 40, 70, 100]


def test_weighted_score_formula():
    s = weighted_score({"opening": np.array([50.0]), "sewer": np.array([60.0]),
                        "rain": np.array([25.0]), "terrain": np.array([70.0])}, C["weights"])
    w = C["weights"]
    assert s[0] == pytest.approx((w["opening"] * 50 + w["sewer"] * 60 + w["rain"] * 25 + w["terrain"] * 70)
                                 / sum(w.values()))


def test_overrides():
    lv = classify(np.array([10.0, 10.0, 10.0, 10.0]), np.array([0.0, 0.05, 0.30, 0.0]), C["levels"], C["override"],
                  rise30=np.array([0.0, 0.0, 0.0, 0.03]), rain60=np.array([0, 0, 0, 60.0]))
    assert list(lv) == [0, 2, 3, 2]   # 그대로 / 개구부 초과 → 경고 / 0.30m → 위험 / 급상승+호우 → 경고


def test_levels():
    assert list(classify(np.array([19.9, 20, 39.9, 40, 69.9, 70]), np.zeros(6), C["levels"], C["override"])) == \
        [0, 1, 1, 2, 2, 3]


# ---- 지표면 수심 보정
def test_surface_depth_needs_heavy_rain_and_water_for_lowland():
    cfg = C["surface"]
    assert surface_depth(0, 20, 1.0, cfg) == 0                     # 이슬비·저지대 → 물 없음
    d = surface_depth(0, cfg["rain_drain_capacity_mmh"] + 30, 1.0, cfg)
    assert d == pytest.approx(cfg["beta"] * 30 + cfg["gamma"])
    assert surface_depth(0.4, 0, 0, cfg) == pytest.approx(cfg["alpha"] * 0.4)


# ---- 하수관 예측
def test_sewer_prediction_trend_and_cap():
    f = {"H0": 1.0, "rise30": 0.02}
    assert predict_sewer(f, 30, 2.0, 1.5) == pytest.approx(1.6)
    assert predict_sewer(f, 60, 1.2, 1.5) == pytest.approx(1.8)    # 상한 1.5 × 1.2
    assert predict_sewer({"H0": 0.2, "rise30": -0.05}, 60, 1.0, 1.5) == 0
    assert predict_sewer({"H0": 0.5, "rise30": None}, 60, 1.0, 1.5) == 0.5


# ---- 알림 시점
HZ = [0, 10, 20, 30, 40, 50, 60]


def _ev(state, t, display, prev, levels, tf=-1):
    return evaluate(state, t, display, prev, levels, HZ, tf, P)


def test_pre_alert_once_then_earlier_resend_then_clear():
    a, st = _ev({}, T0, 1, 1, [1, 1, 1, 1, 2, 2, 3])                       # 40분 뒤 경고 예상
    assert [x["kind"] for x in a] == ["pre"] and a[0]["minutes"] == 40
    a, st = _ev(st, T0 + timedelta(minutes=10), 1, 1, [1, 1, 1, 2, 2, 3, 3])  # 같은 시각(20:40) 예상 → 재발송 없음
    assert a == []
    a, st = _ev(st, T0 + timedelta(minutes=20), 1, 1, [1, 2, 2, 3, 3, 3, 3])  # 20:30으로 10분 앞당겨짐 → 재발송
    assert a and a[0]["reason"] in ("earlier", "higher", "final")
    t = T0 + timedelta(minutes=30)
    for k in range(8):                                                       # 위험 사라진 뒤 60분 → 상태 해제
        a, st = _ev(st, t + timedelta(minutes=10 * k), 0, 0, [0] * 7)
    assert not st["pre_active"]


def test_inflow_eta_triggers_pre_alert_even_if_level_low():
    a, _ = _ev({}, T0, 0, 0, [0, 0, 1, 1, 1, 1, 1], tf=30)
    assert a and a[0]["kind"] == "pre" and a[0]["t_flood"] == 30 and a[0]["eta"].endswith("20:30")


def test_now_alert_on_rise():
    a, st = _ev({}, T0, 2, 1, [2] * 7)
    assert any(x["kind"] == "now" and x["level"] == 2 for x in a)
    a, _ = _ev(st, T0 + timedelta(minutes=10), 2, 2, [2] * 7)    # 그대로면 다시 안 보냄
    assert not any(x["kind"] == "now" for x in a)


def test_predict_sewer_arx_rain_raises_and_cap():
    from src.engine.risk_model import predict_sewer_arx
    coef = [0.9, 0.05, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.05]      # a, b, c0..c5, d
    dry = predict_sewer_arx(coef, 1.0, 1.0, [0] * 6, [0] * 6, 6, 2.0, 1.5)
    wet = predict_sewer_arx(coef, 1.0, 1.0, [10] * 6, [20] * 6, 6, 2.0, 1.5)
    assert len(dry) == 6 and wet[-1] > dry[-1]                     # 비가 오면 더 오름
    assert max(predict_sewer_arx(coef, 2.9, 2.9, [100] * 6, [100] * 6, 6, 2.0, 1.5)) <= 3.0   # 상한 1.5 × 2.0


def test_classify_building_offset():
    """D. 취약한 건물(Δ<0)은 같은 점수에서도 더 높은 등급."""
    lv = classify(np.array([35.0, 35.0]), np.array([0.0, 0.0]), [20, 40, 70], C["override"], offset=np.array([0, -10]))
    assert list(lv) == [1, 2]
