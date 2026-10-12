"""실시간 강우: 서울 열린데이터광장 「서울시 강우량 정보」(ListRainfallService).

응답 필드 (2026-10-07 확인): RF_CD 강우계코드, RF_NM 강우계명, GU_NM 구명, RN_10M 10분 강우량(mm, 문자열),
DATA_CLCT_TM 수집시각("YYYY-MM-DD HH:MM"). 관악구 강우계는 신림P(신림동 빗물펌프장)·관악구청 2곳.

SEOUL_OPEN_API_KEY가 없으면 공개 sample 키를 쓰는데, sample은 최근 5건만 주므로
매 수집분을 SQLite에 쌓아 60분 누적강우를 만든다 (처음 30~60분은 자료가 모자라 "데이터 없음").
"""
import logging
import sqlite3
from datetime import datetime, timedelta

import httpx

from src.alerts.subscriptions import _lock, conn
from src.config import settings

log = logging.getLogger(__name__)
URL = "http://openapi.seoul.go.kr:8088/{key}/json/ListRainfallService/{start}/{end}/{gu}"
GU = "관악구"
GAUGES = ["신림P", "관악구청"]   # 신림동 안의 신림P를 우선, 두 곳 평균 사용
MIN_STEPS_R60 = 4                # 최근 60분 중 10분 자료가 4개 미만이면 R60을 계산하지 않음


def key_label() -> str:
    return "인증키" if settings.seoul_open_api_key else "sample 키(최근 5건 제한)"


def init_table():
    with _lock, conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS rain_obs (gauge TEXT, ts TEXT, mm REAL, fetched_at TEXT,"
                  " PRIMARY KEY(gauge, ts))")


def fetch_and_store() -> int:
    """API 호출 → rain_obs에 저장. 새로 들어온 행 수를 반환. 실패 시 예외."""
    key = settings.seoul_open_api_key or "sample"
    end = 60 if settings.seoul_open_api_key else 5    # 2곳 × 30스텝 = 5시간 분량
    r = httpx.get(URL.format(key=key, start=1, end=end, gu=GU), timeout=15)
    r.raise_for_status()
    body = r.json().get("ListRainfallService")
    if not body or body.get("RESULT", {}).get("CODE") != "INFO-000":
        raise RuntimeError(f"강우 API 오류: {r.text[:200]}")
    now = datetime.now().isoformat(timespec="seconds")
    new = 0
    with _lock, conn() as c:
        for row in body.get("row", []):
            try:
                mm = float(row["RN_10M"])
                ts = datetime.strptime(row["DATA_CLCT_TM"], "%Y-%m-%d %H:%M").isoformat(timespec="minutes")
            except (KeyError, ValueError):
                log.warning("강우 행 해석 실패: %s", row)
                continue
            try:
                c.execute("INSERT INTO rain_obs VALUES(?,?,?,?)", (row["RF_NM"], ts, mm, now))
                new += 1
            except sqlite3.IntegrityError:
                pass
    log.info("강우 수집(%s): %d건 중 신규 %d건", key_label(), len(body.get("row", [])), new)
    return new


def load_series(hours: int = 6) -> tuple[list[tuple[datetime, float]], dict]:
    """10분 강우 시계열 (강우계 평균). 수집시각이 강우계마다 같으므로 시각별로 묶는다."""
    since = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="minutes")
    with _lock, conn() as c:
        rows = c.execute("SELECT gauge, ts, mm FROM rain_obs WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
    by_ts: dict[str, dict[str, float]] = {}
    for g, ts, mm in rows:
        if g in GAUGES:
            by_ts.setdefault(ts, {})[g] = mm
    series = [(datetime.fromisoformat(ts), sum(v.values()) / len(v)) for ts, v in sorted(by_ts.items())]
    last = sorted(by_ts.items())[-1] if by_ts else None
    info = {"gauges": GAUGES, "key": key_label(),
            "last_ts": last[0] if last else None, "last_by_gauge": last[1] if last else {}}
    return series, info


def load_gauges(hours: int = 9) -> dict[str, list[tuple[datetime, float]]]:
    """관악구 강우계별 10분 강우 (자치구 침수예보 기준 판정용, district_gate). 수집시각 13:09 = 13:00~13:09 → 13:10으로."""
    since = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="minutes")
    with _lock, conn() as c:
        rows = c.execute("SELECT gauge, ts, mm FROM rain_obs WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
    out: dict[str, list] = {}
    for g, ts, mm in rows:
        t = datetime.fromisoformat(ts)
        t = t.replace(minute=t.minute // 10 * 10, second=0) + timedelta(minutes=10)
        out.setdefault(g, []).append((t, mm))
    return out


def steps_in_last_hour(series, t: datetime) -> int:
    return sum(1 for ts, _ in series if t - timedelta(minutes=60) < ts <= t)
