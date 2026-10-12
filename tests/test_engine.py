from datetime import datetime, timedelta

import pytest

from src.config import load_params
from src.engine.features import rain_features
from src.engine.rules import Hysteresis, grade_of, rain_hazard, score, vulnerability
from src.sources.replay import load_intensity, to_10min_depths

P = load_params()
T0 = datetime(2022, 8, 8, 18, 0)


def test_rain_hazard_formula():
    # 0.6*95/95 + 0.4*120/120 = 1.0
    assert rain_hazard(P, 95, 120) == pytest.approx(1.0)
    assert rain_hazard(P, 0, 0) == 0
    assert rain_hazard(P, 500, 500) == P["rain"]["h_cap"]


def test_missing_rain_is_not_zero():
    assert rain_hazard(P, None, 10) is None
    assert score(P, None, 0.5) is None
    assert grade_of(P, None) == -1


def test_score_and_grade_boundaries():
    assert score(P, 1.0, 1.0) == pytest.approx(100)
    assert score(P, 1.0, 0.0) == pytest.approx(50)   # 취약도 0이어도 폭우면 절반 남음
    assert score(P, 0.0, 1.0) == 0                    # 비가 없으면 0
    assert [grade_of(P, s) for s in (24.9, 25, 49.9, 50, 74.9, 75)] == [0, 1, 1, 2, 2, 3]


def test_vulnerability_renormalizes_missing_terms():
    # 저지대(3m 낮음 → 1)와 평탄(경사 0 → 1)만 있으면 1
    assert vulnerability(P, True, rel_elev_m=-3, slope_pct=0) == pytest.approx(1)
    assert vulnerability(P, True, rel_elev_m=+5, slope_pct=20) == pytest.approx(0)
    assert vulnerability(P, False, rel_elev_m=-3, slope_pct=0) == 0


def test_hysteresis_up_immediate_down_delayed():
    h = Hysteresis(P)
    assert h.update(1, 80, 0) == 3            # 즉시 상승
    assert h.update(1, 72, 10) == 3           # 75-5=70 이상 → 유지
    assert h.update(1, 60, 20) == 3           # 하한 미만 시작
    assert h.update(1, 60, 40) == 3           # 20분 경과 → 유지
    assert h.update(1, 60, 50) == 2           # 30분 지속 → 한 단계 하향
    assert h.update(1, 90, 51) == 3


def test_rain_features():
    series = [(T0 + timedelta(minutes=10 * i), 5.0) for i in range(1, 10)]
    r60, i10 = rain_features(series, T0 + timedelta(minutes=90))
    assert r60 == pytest.approx(30)
    assert i10 == pytest.approx(30)
    assert rain_features(series, T0) == (None, None)


def test_replay_hyetograph_total_matches_inp_note():
    series = to_10min_depths(load_intensity())
    total = sum(v for _, v in series)
    assert total == pytest.approx(236.4, abs=1.0)    # .inp 주석: 총강우량 236.4mm
    peak = max(series, key=lambda x: x[1])
    assert peak[1] == pytest.approx(130 / 6, abs=0.1)  # 첨두 130mm/hr의 10분 값


def test_insufficient_coverage_is_missing_not_zero():
    series = [(T0 + timedelta(minutes=10 * i), 20.0) for i in (5, 6)]   # 최근 60분 중 2개만
    t = T0 + timedelta(minutes=60)
    assert rain_features(series, t, min_steps=4) == (None, None)
    assert rain_features(series, t)[0] == pytest.approx(40)


def test_kma_rn1_parsing_and_grid():
    from src.sources.kma_api import latlon_to_grid, parse_rn1
    assert latlon_to_grid(37.4842, 126.9297) == (59, 125)
    assert parse_rn1("강수없음") == 0
    assert parse_rn1("1mm 미만") == 0.5
    assert parse_rn1("3.0mm") == 3.0
    assert parse_rn1("30.0~50.0mm") == 40.0
    assert parse_rn1("50.0mm 이상") == 50.0


def test_naver_reverse_parsing():
    from src.geo.naver import parse_reverse
    area = {"area1": {"name": "서울특별시"}, "area2": {"name": "관악구"}, "area3": {"name": "신림동"}}
    data = {"results": [
        {"name": "roadaddr", "region": area, "land": {"name": "문성로26길", "number1": "30", "number2": "",
                                                       "addition0": {"type": "building", "value": "제일빌라"}}},
        {"name": "addr", "region": area, "land": {"type": "1", "number1": "419", "number2": "1"}},
    ]}
    assert parse_reverse(data) == {"road": "서울특별시 관악구 문성로26길 30", "building": "제일빌라",
                                   "jibun": "서울특별시 관악구 신림동 419-1"}
    data["results"][1]["land"]["type"] = "2"
    assert parse_reverse(data)["jibun"].endswith("신림동 산 419-1")
    assert parse_reverse({"results": []}) == {"road": None, "building": None, "jibun": None}
