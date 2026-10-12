"""강우 시계열(10분 강우량)에서 R60, I10 계산."""
from datetime import datetime, timedelta

STALE_MIN = 30  # 서울시 강우 API는 10~20분 늦게 올라오는 경우가 있어 30분까지는 최신값으로 인정


def rain_features(series: list[tuple[datetime, float]], t: datetime, min_steps: int | None = None):
    """t 시점까지의 R60(mm), I10(mm/hr). t 이전 자료가 하나도 없거나,
    min_steps가 주어졌는데 최근 60분의 10분 자료가 그보다 적으면 (None, None) — 과소추정을 0처럼 쓰지 않음."""
    past = [(ts, v) for ts, v in series if ts <= t]
    if not past:
        return None, None
    if min_steps and sum(1 for ts, _ in past if ts > t - timedelta(minutes=60)) < min_steps:
        return None, None
    r60 = sum(v for ts, v in past if ts > t - timedelta(minutes=60))
    last_ts, last_v = past[-1]
    i10 = last_v * 6 if t - last_ts <= timedelta(minutes=STALE_MIN) else None  # 마지막 값이 오래됐으면 결측
    return r60, i10
