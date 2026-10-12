"""서울시 강우량계(스마트수방시스템, 자치구 47~48곳) 10분 강우 이력 → 사건일별 CSV.

출처: 서울 열린데이터광장
  OA-22715 서울시 강우량 데이터 (2006~2025, 관측소별 연도 CSV, 원시자료·미보정)
  OA-1168  서울시 강우량 정보 → 강우량계 위치정보 / OA-22824 강우량계 위치데이터 (위경도·구청·코드)
서울시 침수예보(자치구 발령)가 실제로 쓰는 강우계라 공식 기준을 그대로 재현할 수 있고, 기상청 AWS(61곳)와 합치면
IDW 보간 관측망이 약 2배로 촘촘해진다.

시각: 자료수집시각 13:09 = 13:00:00~13:09:59 강우(OA-1168 설명) → 구간 끝 13:10으로 맞춰 AWS(구간 끝 시각)와 같게.
관측소 번호: 20000 + 강우량계 코드 (AWS 번호와 겹치지 않게)

실행: .venv\\Scripts\\python -m src.calibration.seoul_gauges
"""
import logging
import re
from pathlib import Path

import pandas as pd

from src.calibration import seoul_data as sd

log = logging.getLogger(__name__)
SRC = sd.DATA / "raw" / "seoul_open"
OUT = sd.DATA / "raw" / "rainfall" / "10min_seoul_gauge"
STN_OFFSET = 20000
# 파일 이름(연도별로 표기가 다름) → 위치 목록의 강우량계명
ALIAS = {"공항": "공항동", "도림2": "도림2동", "서소문청사": "서소문", "월곡1동": "상월곡동"}  # 월곡1동 = 2011~2020 상월곡동 계측소 이름


def stations() -> pd.DataFrame:
    loc = pd.read_excel(SRC / "gauge_loc2.xlsx")
    return pd.DataFrame({"stn": STN_OFFSET + loc["강우량계 코드"].astype(int), "name": loc["강우량계명"],
                         "gu": loc["구청명"], "lon": loc["경도"], "lat": loc["위도"]})


def canon(stem: str) -> str:
    base = re.sub(r"\d{4}년?$", "", stem)
    base = base.replace("펌프장", "P")
    base = re.sub(r"P$", "", base)
    return ALIAS.get(base, base)


def _read(f: Path) -> pd.DataFrame:
    raw = f.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return pd.DataFrame()
    lines = [ln for ln in text.splitlines() if "," in ln]
    rows = [ln.replace('"', "").split(",") for ln in lines if not ln.startswith("강우량계명")]
    d = pd.DataFrame([r[:3] for r in rows if len(r) >= 3], columns=["name", "t", "mm"])
    d["t"] = pd.to_datetime(d.t.str.strip(), errors="coerce")
    d["mm"] = pd.to_numeric(d.mm.str.strip(), errors="coerce")
    return d.dropna()


def build():
    st = stations()
    by_name = {}
    for _, r in st.iterrows():
        by_name[re.sub(r"P$", "", r["name"])] = r.stn
    ev = sd.load_events()
    days = sorted({d for s in ev.days for d in s.split()})
    want = set(days)
    parts, miss = [], set()
    for f in sorted(SRC.glob("rain_seq*/*.csv")):
        if f.name.startswith("0_"):
            continue
        key = canon(f.stem)
        if key not in by_name:
            miss.add(key)
            continue
        d = _read(f)
        if d.empty:
            continue
        # 구간 끝 시각으로 (13:09 → 13:10)
        d["ts"] = d.t.dt.floor("10min") + pd.Timedelta(minutes=10)
        d["day"] = (d.ts - pd.Timedelta(minutes=1)).dt.strftime("%Y%m%d")
        d = d[d.day.isin(want)]
        if len(d):
            parts.append(d.assign(stn=by_name[key])[["stn", "ts", "mm", "day"]])
    if miss:
        log.info("위치 정보 없는 강우계(제외): %s", sorted(miss))
    a = pd.concat(parts, ignore_index=True)
    a = a[(a.mm >= 0) & (a.mm <= 60)]                     # 10분 60mm 초과는 계측 오류로 제외 (서울 최대 기록 ~40mm/10분)
    a = a.groupby(["day", "stn", "ts"], as_index=False).mm.max()
    OUT.mkdir(parents=True, exist_ok=True)
    for day, g in a.groupby("day"):
        g.assign(ts=g.ts.dt.strftime("%Y-%m-%dT%H:%M")).rename(columns={"mm": "rn10"})[["stn", "ts", "rn10"]] \
            .to_csv(OUT / f"{day}.csv", index=False)
    st.to_csv(OUT / "stations.csv", index=False, encoding="utf-8-sig")
    log.info("서울시 강우계 %d곳, 사건일 %d/%d일 자료", a.stn.nunique(), a.day.nunique(), len(days))
    return a


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()
