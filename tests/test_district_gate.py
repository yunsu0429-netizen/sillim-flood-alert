"""자치구 침수예보 기준(서울시) + 건물 기준 결합 (risk_model.official_rule, gate_levels, RiskEngine.district_gate)."""
import copy
from datetime import datetime, timedelta

import numpy as np

from src.engine.risk_model import gate_levels, official_rule


def test_seoul2023_hourly_and_15min():
    p = np.zeros(12); p[:6] = 9.5                      # 1시간 57mm → 55mm 초과
    assert official_rule(p, "seoul2023")[0, 5] and not official_rule(p, "seoul2023")[0, 4]
    q = np.zeros(12); q[3] = 14; q[4] = 13             # 15분 ≈ 14 + 6.5 = 20.5mm > 20
    assert official_rule(q, "seoul2023")[0].any()
    r = np.zeros(12); r[3] = 13; r[4] = 13             # 13 + 6.5 = 19.5 → 미충족, 1시간 26mm
    assert not official_rule(r, "seoul2023")[0].any()


def test_seoul2026_needs_72_or_50_plus_90():
    p = np.zeros(24); p[:6] = 10                       # 60mm/h, 3h 60mm → 미충족
    assert not official_rule(p, "seoul2026").any()
    p[6:18] = 3                                        # 3시간 누적 60+36=96, 1시간 최대 60 ≥ 50 시점 필요
    hit = official_rule(p, "seoul2026")[0]
    assert not hit.any() or hit.argmax() >= 5          # 50mm 이상 1시간과 90mm 3시간이 겹쳐야
    q = np.zeros(12); q[:6] = 12.5                     # 75mm/h
    assert official_rule(q, "seoul2026")[0, 5]


def test_gate_caps_at_watch_without_district_warning():
    lv_g, lv = np.array([3, 2, 1, -1, 0]), np.array([2, 2, 0, -1, 1])
    assert gate_levels(lv_g, lv, False).tolist() == [1, 1, 0, -1, 1]
    assert gate_levels(lv_g, lv, True).tolist() == [3, 2, 0, -1, 1]   # 이동된 경계로 주의 미만이어도 원래 '주의' 유지


def test_engine_gate_from_gauges():
    from src.config import load_params
    from src.engine.forecast import RiskEngine
    p = load_params()
    p = copy.deepcopy(p); p["risk_model"]["district_gate"] = {"enabled": True, "rule": "seoul2023", "theta_shift": -10,
                                                              "lookback_min": 180}
    e = RiskEngine(p, None, [])
    t = datetime(2022, 8, 8, 21, 0)
    heavy = {"신림P": [(t - timedelta(minutes=10 * i), 12.0) for i in range(6)]}
    dry = {"신림P": [(t - timedelta(minutes=10 * i), 0.0) for i in range(6)]}
    assert e.district_gate(t, heavy, [])[0]
    assert not e.district_gate(t, dry, []).any()
    fut = [(t + timedelta(minutes=10 * (i + 1)), 12.0) for i in range(6)]   # 예보로 60분 뒤 충족
    g = e.district_gate(t, dry, fut)
    assert not g[0] and g[-1]
    assert e.district_gate(t, None, []) is None
