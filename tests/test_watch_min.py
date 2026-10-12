"""비·물이 없으면 등급을 띄우지 않음 (watch_min) + 평상시 취약 건물 표시 (vulnerable)."""
from datetime import datetime, timedelta

import numpy as np

from src.engine.risk_model import apply_watch_min

WM = {"rain60_mm": 3, "sewer_util": 0.7}


def test_no_rain_no_water_caps_to_low():
    lv = np.array([1, 2, 3, 0, -1])
    out = apply_watch_min(lv, np.zeros(5), np.full(5, 0.3), np.zeros(5), WM)
    assert out.tolist() == [0, 0, 0, 0, -1]


def test_any_condition_keeps_level():
    lv = np.array([1, 1, 1])
    out = apply_watch_min(lv, np.array([3.0, 0, 0]), np.array([0.2, 0.75, 0.2]), np.array([0, 0, 0.05]), WM)
    assert out.tolist() == [1, 1, 1]


def test_engine_dry_day_has_no_watch_but_marks_vulnerable():
    from src.config import load_params
    from src.engine.forecast import RiskEngine
    e = RiskEngine(load_params(), None, [])
    t = datetime(2026, 10, 12, 11, 0)
    dry = [(t - timedelta(minutes=10 * i), 0.0) for i in range(36)]
    s = e.compute(t, dry, dry, {}, 100, live=False)
    bj = e.banjiha
    assert int((s.grades[bj] >= 1).sum()) == 0                 # 맑은 날: 주의 이상 없음
    n = int(e.vulnerable.sum())
    assert 0.08 * bj.sum() <= n <= 0.12 * bj.sum()               # 반지하 후보 상위 약 10%
    assert not e.vulnerable[~bj].any()
