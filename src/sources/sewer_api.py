"""하수관 수위: 서울 열린데이터광장 「서울시 하수관로 수위 현황」(DrainpipeMonitoringInfo).

응답 필드 (2026-10-07 확인): UNQ_NO 수위계 고유번호(예 21-0001), SE_CD 구분코드(관악=21), SE_NM, MSRMT_YMD 측정시각,
MSRMT_WATL 측정수위(m), SGN_STTS 통신상태, PSTN_INFO 설치 위치(주소 문장, HTML 엔티티 포함).
요청 형식: /{KEY}/json/DrainpipeMonitoringInfo/{시작}/{끝}/{구분코드}/{YYYYMMDDHH 부터}/{YYYYMMDDHH 까지}
- 관악구 수위계 34곳. 11곳은 1분 간격, 나머지는 10분~1시간 간격
- 2022-08-08 자료는 21-0001~21-0007 일곱 곳만 있음 (1분 간격)
이 자료는 하수관로 수위다 (하천 수위와 다름).

명령:
  python -m src.sources.sewer_api --stations          수위계 위치를 네이버 지오코딩해 data/manual/sewer_stations.csv 저장
  python -m src.sources.sewer_api --backfill 2022-08-08 그날 18시~다음날 03시 수위를 data/replay/sewer_20220808.csv 저장
"""
import argparse
import csv
import html
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta

import httpx

from src.alerts.subscriptions import _lock, conn
from src.config import DATA, MANUAL, settings

log = logging.getLogger(__name__)
URL = "http://openapi.seoul.go.kr:8088/{key}/json/DrainpipeMonitoringInfo/{a}/{b}/{gu}/{s}/{e}"
GU_CODE = "21"   # 관악
STATIONS_CSV = MANUAL / "sewer_stations.csv"
REPLAY_CSV = DATA / "replay" / "sewer_20220808.csv"


def init_table():
    with _lock, conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS sewer_obs (station TEXT, ts TEXT, level_m REAL, status TEXT,"
                  " PRIMARY KEY(station, ts))")


def fetch(start: datetime, end: datetime, gu: str = GU_CODE) -> list[dict]:
    """[start 시, end 시] 범위 한 구(기본 관악)의 수위 전부 (1,000건씩 페이지). 실패 시 예외."""
    if not settings.seoul_open_api_key:
        raise RuntimeError("SEOUL_OPEN_API_KEY 없음")
    rows, a = [], 1
    while True:
        url = URL.format(key=settings.seoul_open_api_key, a=a, b=a + 999, gu=gu,
                         s=start.strftime("%Y%m%d%H"), e=end.strftime("%Y%m%d%H"))
        from src.sources.seoul_api import get as seoul_get
        r = seoul_get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        body = data.get("DrainpipeMonitoringInfo")
        if body is None:
            code = data.get("RESULT", {}).get("CODE")
            if code == "INFO-200":   # 해당 데이터 없음
                return rows
            raise RuntimeError(f"하수관 수위 API 오류: {str(data)[:200]}")
        page = body.get("row", [])
        rows += page
        if len(page) < 1000:
            return rows
        a += 1000
        time.sleep(0.1)


def parse(row: dict) -> tuple[str, str, float, str] | None:
    try:
        ts = datetime.strptime(row["MSRMT_YMD"][:16], "%Y-%m-%d %H:%M").isoformat(timespec="minutes")
        return row["UNQ_NO"], ts, float(row["MSRMT_WATL"]), row.get("SGN_STTS") or ""
    except (KeyError, ValueError, TypeError):
        log.warning("하수관 수위 행 해석 실패: %s", row)
        return None


def store(rows: list[dict]) -> int:
    new = 0
    with _lock, conn() as c:
        for row in rows:
            p = parse(row)
            if not p:
                continue
            try:
                c.execute("INSERT INTO sewer_obs VALUES(?,?,?,?)", p)
                new += 1
            except sqlite3.IntegrityError:
                pass
    return new


def fetch_recent(minutes: int = 40) -> int:
    """실시간: 최근 minutes분(시 단위 조회라 앞 시각 포함) 수집 → DB. 신규 행 수."""
    now = datetime.now()
    rows = fetch(now - timedelta(minutes=minutes), now)
    new = store(rows)
    log.info("하수관 수위 수집: %d건 중 신규 %d건", len(rows), new)
    return new


def load_obs(since: datetime) -> dict[str, list[tuple[datetime, float, str]]]:
    with _lock, conn() as c:
        rows = c.execute("SELECT station, ts, level_m, status FROM sewer_obs WHERE ts >= ? ORDER BY ts",
                         (since.isoformat(timespec="minutes"),)).fetchall()
    out: dict[str, list] = {}
    for st, ts, lv, status in rows:
        out.setdefault(st, []).append((datetime.fromisoformat(ts), lv, status))
    return out


def load_replay() -> dict[str, list[tuple[datetime, float, str]]]:
    out: dict[str, list] = {}
    with open(REPLAY_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["station"], []).append((datetime.fromisoformat(r["ts"]), float(r["level_m"]), r["status"]))
    for v in out.values():
        v.sort()
    return out


def load_stations() -> list[dict]:
    if not STATIONS_CSV.exists():
        log.warning("%s 없음 → 하수관 요소 없이 계산 (python -m src.sources.sewer_api --stations)", STATIONS_CSV.name)
        return []
    with open(STATIONS_CSV, encoding="utf-8") as f:
        return [r | {"lng": float(r["lng"]), "lat": float(r["lat"])} for r in csv.DictReader(f) if r.get("lng")]


def clean_address(pstn: str) -> str:
    """'관악구 원신길 11옆 맨홀<동방빌딩 옆, ...>' → '서울특별시 관악구 원신길 11'"""
    s = html.unescape(pstn).replace("\n", " ")
    s = re.split(r"[<(（]", s)[0]
    s = re.sub(r"(앞|옆|인근|사거리|맨홀|작업구|하수박스|천장|도로|보도|교차로|주소).*$", "", s).strip()
    # '신림로71길 45', '봉천로4길 9', '난곡로351', '조원로 66' 모두 도로명 + 건물번호까지 잘라냄
    m = re.match(r"(관악구\s+[가-힣]+(?:로|길)(?:\s*\d+(?:번)?길)?\s*\d+(?:-\d+)?)", s)
    s = m.group(1) if m else s
    return s if s.startswith("서울") else f"서울특별시 {s}"


def build_stations():
    from src.geo import naver
    rows = fetch(datetime.now() - timedelta(hours=6), datetime.now())
    seen = {}
    for r in rows:
        seen.setdefault(r["UNQ_NO"], html.unescape(r["PSTN_INFO"]).replace("\n", " ").strip())
    out = []
    for sid, pstn in sorted(seen.items()):
        q = clean_address(pstn)
        hit = None
        try:
            res = naver.geocode(q)
            hit = res[0] if res else None
        except Exception as e:
            log.warning("%s 지오코딩 실패: %s", sid, e)
        if not hit or "관악구" not in (hit.get("address") or hit.get("road") or ""):
            log.warning("%s 위치 확인 필요: '%s' → %s", sid, q, hit)
        out.append({"station": sid, "pstn_info": pstn, "query": q,
                    "geocoded": (hit or {}).get("road") or (hit or {}).get("address") or "",
                    "lng": round(hit["lng"], 6) if hit else "", "lat": round(hit["lat"], 6) if hit else ""})
    STATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(STATIONS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    log.info("수위계 %d곳 저장 (좌표 있음 %d곳) → %s", len(out), sum(1 for r in out if r["lng"]), STATIONS_CSV)


def backfill(day: str):
    d = datetime.strptime(day, "%Y-%m-%d")
    rows = fetch(d.replace(hour=18), d + timedelta(hours=27))
    parsed = sorted(p for p in map(parse, rows) if p)
    REPLAY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REPLAY_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "ts", "level_m", "status"])
        w.writerows(parsed)
    log.info("%s 하수관 수위 %d행 (수위계 %d곳) → %s", day, len(parsed), len({p[0] for p in parsed}), REPLAY_CSV)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", action="store_true")
    ap.add_argument("--backfill")
    a = ap.parse_args()
    if a.stations:
        build_stations()
    if a.backfill:
        backfill(a.backfill)
