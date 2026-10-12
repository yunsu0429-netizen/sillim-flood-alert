"""v0.2 엔진 전체 경로(ARX 하수관 예측·SWMM·건물별 경계 조정)가 한 주기 계산되는지."""
from datetime import datetime, timedelta

import numpy as np

from src.config import load_params
from src.engine.forecast import RiskEngine
from src.sources import sewer_api


def test_engine_compute_v02_runs_and_heavy_rain_raises_levels():
    e = RiskEngine(load_params(), None, sewer_api.load_stations())
    t = datetime(2022, 8, 8, 21, 0)
    dry = [(t - timedelta(minutes=10 * i), 0.0) for i in range(36)]
    wet = [(t - timedelta(minutes=10 * i), 20.0) for i in range(36)]
    fut = [(t + timedelta(minutes=10 * i), 20.0) for i in range(1, 7)]
    s_dry = e.compute(t, dry, dry, {}, 100, live=False)
    e.reset_state()
    s_wet = e.compute(t, wet, wet + fut, {}, 100, live=False)
    b = e.banjiha
    assert (s_wet.levels[:, b] >= s_dry.levels[:, b]).all()
    assert (s_wet.levels[0, b] >= 2).mean() > (s_dry.levels[0, b] >= 2).mean()
    assert np.isfinite(e.model.delta).all()
