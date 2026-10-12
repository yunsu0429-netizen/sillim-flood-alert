"""기상청 초단기예보(getUltraSrtFcst) — 향후 6시간 1시간 강수량(RN1).

매시 30분 발표, 약 45분 뒤 조회 가능. 신림동(37.4842N, 126.9297E) = 격자 (59, 125).
RN1은 범주형 문자열("강수없음", "1mm 미만", "3.0mm", "30.0~50.0mm", "50.0mm 이상")이라 대표값으로 바꾼다:
미만 → 0.5mm, 구간 → 중간값, 이상 → 하한값. fcstTime의 RN1은 그 시각 직전 1시간 강수량으로 보고
1시간을 10분 6칸에 균등 배분한다 (가정).
"""
import logging
import math
import re
from datetime import datetime, timedelta

import httpx

from src.config import settings

log = logging.getLogger(__name__)
URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
LAT, LON = 37.4842, 126.9297


def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """기상청 동네예보 격자 변환 (Lambert Conformal Conic, 5km)."""
    re_, grid, s1, s2, olon, olat, xo, yo = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    d = math.pi / 180
    re_ /= grid
    s1, s2, olon, olat = s1 * d, s2 * d, olon * d, olat * d
    sn = math.log(math.cos(s1) / math.cos(s2)) / math.log(math.tan(math.pi / 4 + s2 / 2) / math.tan(math.pi / 4 + s1 / 2))
    sf = math.tan(math.pi / 4 + s1 / 2) ** sn * math.cos(s1) / sn
    ro = re_ * sf / math.tan(math.pi / 4 + olat / 2) ** sn
    ra = re_ * sf / math.tan(math.pi / 4 + lat * d / 2) ** sn
    th = lon * d - olon
    th = th - 2 * math.pi if th > math.pi else th + 2 * math.pi if th < -math.pi else th
    th *= sn
    return int(ra * math.sin(th) + xo + 0.5), int(ro - ra * math.cos(th) + yo + 0.5)


def base_time(now: datetime) -> datetime:
    """조회 가능한 가장 최근 발표 시각 (HH:30, 45분 지연 반영)."""
    t = now - timedelta(minutes=45)
    t = t.replace(minute=30, second=0, microsecond=0)
    return t if t <= now - timedelta(minutes=45) else t - timedelta(hours=1)


def parse_rn1(v: str) -> float:
    v = v.strip()
    if v in ("강수없음", "-", "0"):
        return 0.0
    if "미만" in v:
        return 0.5
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", v)]
    if not nums:
        raise ValueError(f"RN1 해석 불가: {v}")
    if "~" in v and len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    return nums[0]


def fetch_forecast(now: datetime | None = None) -> tuple[list[tuple[datetime, float]], dict]:
    """(10분 단위 예측 강우 시계열, 정보). 키가 없으면 ([], 정보)."""
    now = now or datetime.now()
    if not settings.kma_api_key:
        return [], {"error": "KMA_API_KEY 없음"}
    nx, ny = latlon_to_grid(LAT, LON)
    base = base_time(now)
    params = {"serviceKey": settings.kma_api_key, "numOfRows": 60, "pageNo": 1, "dataType": "JSON",
              "base_date": base.strftime("%Y%m%d"), "base_time": base.strftime("%H%M"), "nx": nx, "ny": ny}
    r = httpx.get(URL, params=params, timeout=20)
    r.raise_for_status()
    body = r.json()["response"]
    if body["header"]["resultCode"] != "00":
        raise RuntimeError(f"기상청 API 오류: {body['header']}")
    hourly = []
    for it in body["body"]["items"]["item"]:
        if it["category"] == "RN1":
            end = datetime.strptime(it["fcstDate"] + it["fcstTime"], "%Y%m%d%H%M")
            hourly.append((end, parse_rn1(it["fcstValue"]), it["fcstValue"]))
    hourly.sort()
    series = []
    for end, mm, _ in hourly:
        for k in range(5, -1, -1):
            series.append((end - timedelta(minutes=10 * k), mm / 6))
    info = {"base": base.strftime("%m/%d %H:%M"), "grid": [nx, ny],
            "hourly": [(e.strftime("%H:%M"), raw) for e, _, raw in hourly]}
    log.info("기상청 초단기예보 %s 발표, RN1 %s", info["base"], info["hourly"])
    return series, info


def merge(observed: list[tuple[datetime, float]], forecast: list[tuple[datetime, float]]):
    """관측 이후 시각만 예보로 채운 시계열 (관측 우선)."""
    last = observed[-1][0] if observed else datetime.min
    return observed + [(ts, v) for ts, v in forecast if ts > last]
