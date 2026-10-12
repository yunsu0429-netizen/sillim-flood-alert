"""기상청 API허브 AWS 자료 (칼리브레이션용 과거 강수).

- 매분자료(nph-aws2_min)는 관측소를 지정하면 응답이 없고, stn=0(전체)으로 한 시각씩만 조회된다.
  RN-DAY(그 시각까지 일강수, 00시 기준 누적)를 10분 간격으로 받아 차분해 10분 강수를 만든다.
- 비 오는 시간만 받기 위해 먼저 1시간 간격으로 훑고, 시간 강수가 있는 구간 ±1시간만 10분 간격으로 채운다.
결과는 data/raw/rainfall/10min/{YYYYMMDD}.csv (stn, ts, rn10) 로 캐시한다.
"""
import asyncio
import csv
import logging
from datetime import datetime, timedelta

import httpx

from src.config import DATA, settings

log = logging.getLogger(__name__)
URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
STATIONS = {410: "기상청", 417: "금천", 509: "관악"}     # 신림동에서 1.9 / 2.7 / 3.5 km
CACHE = DATA / "raw" / "rainfall" / "10min"


async def _snapshot(client, sem, tm: datetime, stations=STATIONS) -> dict[int, float]:
    """한 시각 전체 관측소 → {stn: RN-DAY} (stations에 든 것만). 결측(-99 이하)은 제외."""
    async with sem:
        for attempt in range(4):
            try:
                r = await client.get(URL, params={"tm2": tm.strftime("%Y%m%d%H%M"), "stn": "0", "disp": "1",
                                                  "help": "0", "authKey": settings.kma_apihub_key}, timeout=60)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"AWS 매분자료 실패 {tm}")
    out = {}
    for line in r.content.decode("euc-kr", "replace").splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split(",")
        try:
            stn, rn_day = int(p[1]), float(p[13])
        except (IndexError, ValueError):
            continue
        if stn in stations and rn_day > -90:
            out[stn] = rn_day
    return out


async def fetch_day(day: str, concurrency: int = 6, stations=STATIONS) -> list[tuple[int, datetime, float]]:
    """day(YYYYMMDD)의 10분 강수 [(stn, 구간끝 시각, mm)]. 비 없는 시간대는 그 시간 강수(0)를 고르게 나눈다."""
    d0 = datetime.strptime(day, "%Y%m%d")
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        hours = [d0 + timedelta(hours=h) for h in range(0, 25)]
        hs = await asyncio.gather(*[_snapshot(client, sem, t, stations) for t in hours])
        wet = set()   # 시간 강수 > 0 인 시간과 그 앞뒤를 10분 간격으로
        for i in range(1, len(hours)):
            for stn in stations:
                a, b = hs[i - 1].get(stn), hs[i].get(stn)
                if a is not None and b is not None and (b - a if i > 1 else b) > 0:
                    wet.update({max(1, i - 1), i, min(24, i + 1)})
        need = sorted({hours[i - 1] + timedelta(minutes=m) for i in wet for m in range(10, 60, 10)})
        snaps = dict(zip(hours, hs))
        missing = [t for t in need if t not in snaps]
        got = await asyncio.gather(*[_snapshot(client, sem, t, stations) for t in missing])
        snaps.update(zip(missing, got))
    times = sorted(snaps)
    out = []
    for stn in stations:
        prev_t, prev_v = None, None
        for t in times:
            v = snaps[t].get(stn)
            if v is None:
                continue
            if t == d0:
                prev_t, prev_v = t, 0.0          # 00:00 값은 전날 합계 → 기준을 0으로
                continue
            if prev_v is not None:
                inc = v - prev_v if v >= prev_v else v
                steps = max(1, round((t - prev_t).total_seconds() / 600))
                for s in range(steps):
                    out.append((stn, prev_t + timedelta(minutes=10 * (s + 1)), round(inc / steps, 2)))
            prev_t, prev_v = t, v
    return out


def load_day(day: str, cache=CACHE, stations=STATIONS) -> list[tuple[int, datetime, float]]:
    path = cache / f"{day}.csv"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return [(int(r["stn"]), datetime.fromisoformat(r["ts"]), float(r["rn10"])) for r in csv.DictReader(f)]
    rows = asyncio.run(fetch_day(day, stations=stations))
    cache.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stn", "ts", "rn10"])
        w.writerows([(s, t.isoformat(timespec="minutes"), v) for s, t, v in rows])
    return rows
