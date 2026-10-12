"""서울 전역 확장 보정용 자료 준비.

신림동만으로는 침수 사건이 3개(하수관 실측이 있는 사건 1개)뿐이라 계수를 정할 수 없어서,
같은 자료원이 서울 전역에 있다는 점을 이용해 보정 표본을 넓힌다.

  daily        기상청 AWS 일강수 2010~2025 6~9월, 서울 일대 관측소 좌표 포함 (sfc_aws_day, stn=0)
  events       호우일(서울 관측소 최대 일강수 ≥ 80mm) + 침수흔적도 날짜 → 사건 목록
  rain10       사건일 서울 관측소 전부의 10분 강수 (kma_aws, RN-DAY 차분)
  sewer        서울 25개 구 하수관 수위계 위치(네이버 지오코딩) + 2020년 이후 사건일 수위
  buildings    서울 반지하 후보(주거 + 지하층 ≥ 1) + 상대고도·경사 (DEM이 덮는 곳만)
  labels       침수흔적도 × 건물 공간조인 (겹친 비율 ≥ 0.1)

결과: data/calibration/seoul/ , 원자료 캐시: data/raw/rainfall/10min_seoul/, data/raw/sewer_seoul/
실행: .venv\\Scripts\\python -m src.calibration.seoul_data <단계...>
"""
import argparse
import asyncio
import calendar
import csv
import glob
import html
import logging
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
import pyogrio

from src.calibration import kma_aws
from src.config import DATA, settings
from src.sources import sewer_api

log = logging.getLogger(__name__)
OUT = DATA / "calibration" / "seoul"
RAIN10 = DATA / "raw" / "rainfall" / "10min_seoul"
SEWER = DATA / "raw" / "sewer_seoul"
TRACE = DATA / "raw" / "flood_trace"
BBOX = (126.70, 37.38, 127.25, 37.74)        # 서울 + 경계 밖 2~3km (IDW 가장자리 보완)
HEAVY_MM = 80.0
SEWER_FROM = "20200601"                      # 하수관 수위 이력은 2020년부터
RESIDENTIAL = ("01000", "02000")


# ---------------------------------------------------------------- 강우
def daily():
    rows = []
    with httpx.Client(timeout=120) as c:
        for y in range(2010, 2026):
            for m in range(6, 10):
                last = calendar.monthrange(y, m)[1]
                r = c.get("https://apihub.kma.go.kr/api/typ01/url/sfc_aws_day.php",
                          params={"tm1": f"{y}{m:02d}01", "tm2": f"{y}{m:02d}{last}", "obs": "rn_day", "stn": "0",
                                  "disp": "1", "help": "0", "authKey": settings.kma_apihub_key})
                for line in r.content.decode("euc-kr", "replace").splitlines():
                    if not line or line.startswith("#"):
                        continue
                    p = line.rstrip("=").split(",")
                    try:
                        lon, lat = float(p[2]), float(p[3])
                    except (IndexError, ValueError):
                        continue
                    if BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]:
                        rows.append((p[0], int(p[1]), lon, lat, float(p[4]), float(p[5]), p[6] if len(p) > 6 else ""))
            log.info("일강수 %d년 누적 %d행", y, len(rows))
    d = pd.DataFrame(rows, columns=["date", "stn", "lon", "lat", "ht", "rn_day", "name"])
    OUT.mkdir(parents=True, exist_ok=True)
    st = d.sort_values("date").groupby("stn").last()[["lon", "lat", "ht", "name"]].reset_index()
    st.to_csv(OUT / "aws_stations.csv", index=False, encoding="utf-8")
    d[["date", "stn", "rn_day"]].to_csv(OUT / "aws_daily.csv", index=False, encoding="utf-8")
    log.info("관측소 %d곳, 일강수 %d행 저장", len(st), len(d))


def stations() -> pd.DataFrame:
    return pd.read_csv(OUT / "aws_stations.csv")


def trace_dates() -> pd.DataFrame:
    """침수흔적도 전체 → (연도, 날짜, 폴리곤 수). 2025년 파일은 칼럼 구조가 다름(피해일시)."""
    out = []
    for y in sorted(glob.glob(str(TRACE / "*"))):
        d = load_trace(y[-4:], geometry=False)
        out.append(d.groupby("date").size().rename("n").reset_index().assign(year=y[-4:]))
    return pd.concat(out, ignore_index=True)


def load_trace(year: str, geometry=True) -> pd.DataFrame:
    f = glob.glob(str(TRACE / year / "*.shp"))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pyogrio.read_dataframe(f, encoding="cp949", read_geometry=geometry)
    col = "F_SAT_YMD" if "F_SAT_YMD" in d else "피해일시"
    d["date"] = d[col].astype(str).str.replace("-", "").str[:8]
    d["time"] = d[col].astype(str).str.extract(r"-\s*(\d{4})")[0] if col == "피해일시" else None
    kind = d["F_DISA_NM"].astype(str) if "F_DISA_NM" in d else d.get("TYPE2", pd.Series("", index=d.index)).astype(str)
    d["label_kind"] = np.where(kind.str.contains("지원금|보험"), "신고", "조사")
    if geometry:
        d = d.to_crs(5186)
        d = d[d.geometry.notna() & d.geometry.is_valid]
    return d


def events() -> pd.DataFrame:
    """호우일(서울 관측소 최대 일강수 ≥ 80mm)을 연속일로 묶어 사건으로 하고, 침수흔적 날짜는 가까운 사건에 붙인다.
    흔적 날짜는 신고·조사 날짜라 호우 뒤 며칠씩 이어지므로 사건 기간을 늘리지 않는다:
    사건 기간 ±1일 → 그 사건, 아니면 1~3일 뒤 시작하는 사건(호우 첫날로 기록된 경우), 둘 다 없으면 '짝 없는 흔적'으로 제외."""
    dd = pd.read_csv(OUT / "aws_daily.csv", dtype={"date": str})
    dd = dd[dd.rn_day >= 0]
    mx = dd.groupby("date").rn_day.max()
    heavy = sorted(mx[mx >= HEAVY_MM].index)
    ev, cur = [], []
    for d in heavy:
        if cur and (datetime.strptime(d, "%Y%m%d") - datetime.strptime(cur[-1], "%Y%m%d")).days <= 1:
            cur.append(d)
        else:
            if cur:
                ev.append(cur)
            cur = [d]
    ev.append(cur)
    dt = lambda x: datetime.strptime(x, "%Y%m%d")
    td = trace_dates()
    td = td[td.date.str.match(r"20\d{6}$")].groupby("date").n.sum()
    attach = {i: {} for i in range(len(ev))}
    orphan = {}
    for d, n in td.items():
        hit = None
        for i, e in enumerate(ev):
            if dt(e[0]) - timedelta(days=1) <= dt(d) <= dt(e[-1]) + timedelta(days=1):
                hit = i
                break
        if hit is None:
            for i, e in enumerate(ev):
                if timedelta(days=0) < dt(e[0]) - dt(d) <= timedelta(days=3):
                    hit = i
                    break
        if hit is None:
            orphan[d] = int(n)
        else:
            attach[hit][d] = int(n)
    # 짝 없는 흔적 중 사건 끝 뒤 20일 안 = 늦게 접수된 신고로 보고, 그 사건에서 정답·오답 어느 쪽에도 쓰지 않는다
    late = {i: [] for i in range(len(ev))}
    for d in list(orphan):
        prev = [i for i, e in enumerate(ev) if timedelta(days=1) < dt(d) - dt(e[-1]) <= timedelta(days=20)]
        if prev:                                   # 가장 가까운(마지막) 앞 사건
            late[max(prev, key=lambda i: dt(ev[i][-1]))].append(d)
    rows = [{"event": f"{e[0][:4]}-{e[0][4:6]}-{e[0][6:]}", "days": " ".join(e), "n_days": len(e),
             "max_daily_mm": float(mx.reindex(e).max()), "trace_dates": " ".join(sorted(attach[i])),
             "trace_polys": sum(attach[i].values()), "late_dates": " ".join(late[i]),
             "sewer": e[0] >= SEWER_FROM} for i, e in enumerate(ev)]
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "events.csv", index=False, encoding="utf-8")
    pd.Series(orphan, name="polys").rename_axis("date").to_csv(OUT / "trace_orphans.csv", encoding="utf-8")
    log.info("사건 %d개 (침수흔적 있음 %d, 하수관 기간 %d), 짝 없는 흔적 날짜 %d개(폴리곤 %d)", len(d),
             int((d.trace_polys > 0).sum()), int(d.sewer.sum()), len(orphan), sum(orphan.values()))
    return d


def load_events() -> pd.DataFrame:
    return pd.read_csv(OUT / "events.csv")


def rain10(concurrency=4):
    st = set(stations().stn.astype(int))
    ev = load_events()
    # 우선순위: 하수관 기간 침수 사건 → 큰 침수 사건 → 하수관 기간 비침수 → 나머지
    ev["prio"] = np.select([ev.sewer & (ev.trace_polys > 0), ev.trace_polys >= 100, ev.sewer], [0, 1, 2], 3)
    days = [d for _, e in ev.sort_values(["prio", "event"]).iterrows() for d in e.days.split()]
    days = list(dict.fromkeys(days))
    todo = [d for d in days if not (RAIN10 / f"{d}.csv").exists()]
    log.info("10분 강수: %d일 중 %d일 수집 (관측소 %d곳)", len(days), len(todo), len(st))
    for i, d in enumerate(todo):
        try:
            kma_aws.load_day(d, cache=RAIN10, stations=st)
        except Exception as e:
            log.warning("%s 실패: %s", d, e)
        log.info("10분 강수 %d/%d %s", i + 1, len(todo), d)


# ---------------------------------------------------------------- 하수관
def _clean(pstn: str) -> str:
    s = html.unescape(pstn).replace("\n", " ")
    s = re.split(r"[<(（]", s)[0]
    s = re.sub(r"(앞|옆|인근|사거리|맨홀|작업구|하수박스|천장|도로|보도|교차로|주소).*$", "", s).strip()
    m = re.match(r"([가-힣]+구\s+[가-힣0-9]+(?:로|길)(?:\s*\d+(?:번)?길)?\s*\d+(?:-\d+)?)", s)
    s = m.group(1) if m else s
    return s if s.startswith("서울") else f"서울특별시 {s}"


def sewer_stations():
    from src.geo import naver
    seen = {}
    probes = [datetime.now() - timedelta(hours=3), datetime(2022, 8, 8, 21), datetime(2020, 8, 1, 12)]
    for g in range(1, 26):
        for t in probes:
            try:
                for r in sewer_api.fetch(t, t, gu=f"{g:02d}"):
                    seen.setdefault(r["UNQ_NO"], (r.get("SE_NM", ""), html.unescape(r["PSTN_INFO"]).replace("\n", " ")))
            except Exception as e:
                log.warning("구 %02d %s 조회 실패: %s", g, t, e)
    out = []
    for sid, (gu, pstn) in sorted(seen.items()):
        q = _clean(pstn)
        hit = None
        try:
            res = naver.geocode(q)
            hit = res[0] if res else None
        except Exception as e:
            log.warning("%s 지오코딩 실패: %s", sid, e)
        out.append({"station": sid, "gu": gu, "pstn_info": pstn, "query": q,
                    "geocoded": (hit or {}).get("road") or (hit or {}).get("address") or "",
                    "lng": round(hit["lng"], 6) if hit else "", "lat": round(hit["lat"], 6) if hit else ""})
    pd.DataFrame(out).to_csv(OUT / "sewer_stations.csv", index=False, encoding="utf-8")
    log.info("수위계 %d곳 (좌표 있음 %d곳)", len(out), sum(1 for r in out if r["lng"] != ""))


def _sewer_day(day: str) -> int:
    path = SEWER / f"{day}.csv"
    if path.exists():
        return -1
    d0 = datetime.strptime(day, "%Y%m%d")
    rows = []
    for g in range(1, 26):
        for attempt in range(3):
            try:
                rows += sewer_api.fetch(d0, d0 + timedelta(hours=23), gu=f"{g:02d}")
                break
            except Exception as e:
                log.warning("%s 구 %02d 재시도 %d: %s", day, g, attempt + 1, e)
    parsed = sorted(p for p in map(sewer_api.parse, rows) if p)
    SEWER.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "ts", "level_m", "status"])
        w.writerows(parsed)
    return len(parsed)


def sewer_history(workers=4):
    days = sorted({d for s in load_events().days for d in s.split() if d >= SEWER_FROM})
    log.info("하수관 수위: 사건일 %d일", len(days))
    with ThreadPoolExecutor(workers) as ex:
        for d, n in zip(days, ex.map(_sewer_day, days)):
            log.info("하수관 %s: %s행", d, "캐시" if n < 0 else n)


# ---------------------------------------------------------------- 건물·라벨
def buildings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = pyogrio.read_dataframe(settings.building_shp, columns=["A1", "A3", "A8", "A11", "A12", "A13", "A26", "A27"],
                                   encoding="cp949")
    floors_down = pd.to_numeric(b.A27, errors="coerce").fillna(0)
    b = b[b.A8.astype(str).isin(RESIDENTIAL) & (floors_down >= 1)].copy()
    b = b[b.geometry.notna()].reset_index(drop=True)
    b["bid"] = np.arange(len(b))
    b = b.rename(columns={"A1": "gid", "A3": "bjd", "A11": "structure", "A12": "bldg_area_m2", "A13": "approved",
                          "A26": "floors_up", "A27": "floors_down"})
    for c in ("bldg_area_m2", "floors_up", "floors_down"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["sgg"] = b.bjd.astype(str).str[:5]
    pts = b.geometry.representative_point()
    b["x"], b["y"] = pts.x, pts.y
    b["rel_elev_m"], b["slope_pct"] = _terrain(b.x.to_numpy(), b.y.to_numpy())
    b["rel_elev_100m"], _ = _terrain(b.x.to_numpy(), b.y.to_numpy(), radius_m=100)
    b["rel_elev_1km"], _ = _terrain(b.x.to_numpy(), b.y.to_numpy(), radius_m=1000)
    b = b[["bid", "gid", "bjd", "sgg", "approved", "structure", "bldg_area_m2", "floors_up", "floors_down", "x", "y",
           "rel_elev_m", "rel_elev_100m", "rel_elev_1km", "slope_pct", "geometry"]]
    b.to_parquet(OUT / "buildings.parquet")
    log.info("반지하 후보 %d동 (지형값 있음 %d동)", len(b), int(b.rel_elev_m.notna().sum()))


def _terrain(x, y, radius_m=300):
    """DEM으로 상대고도(주변 radius 평균 대비)와 경사(%). DEM 밖은 NaN. DEM_DIR(5m)이 있으면 그것을 쓴다."""
    import pyproj
    import rasterio
    from scipy import ndimage
    mosaic = DATA / "raw" / "dem" / "seoul_dem90_mosaic.tif"   # 공개DEM 37608·37612·37705·37709 병합
    path = str(mosaic) if mosaic.exists() else settings.dem_path
    with rasterio.open(path) as r:
        z = r.read(1).astype(float)
        nd = r.nodata
        tr, crs = r.transform, r.crs
    if nd is not None:
        z[z == nd] = np.nan
    res = abs(tr.a)
    k = max(3, int(round(2 * radius_m / res)) | 1)
    valid = ~np.isnan(z)
    zf = np.nan_to_num(z)
    mean = ndimage.uniform_filter(zf, k) / np.maximum(ndimage.uniform_filter(valid.astype(float), k), 1e-9)
    gy, gx = np.gradient(zf, res)
    slope = np.hypot(gx, gy) * 100
    t = pyproj.Transformer.from_crs(5186, crs, always_xy=True)
    xx, yy = t.transform(x, y)
    col = ((np.asarray(xx) - tr.c) / tr.a).astype(int)
    row = ((np.asarray(yy) - tr.f) / tr.e).astype(int)
    ok = (row >= 0) & (row < z.shape[0]) & (col >= 0) & (col < z.shape[1])
    rel, slp = np.full(len(x), np.nan), np.full(len(x), np.nan)
    rr, cc = row[ok], col[ok]
    v = valid[rr, cc]
    rel[np.flatnonzero(ok)[v]] = (z - mean)[rr[v], cc[v]]
    slp[np.flatnonzero(ok)[v]] = slope[rr[v], cc[v]]
    return rel, slp


def _match(b, area, tr, min_ratio):
    tr = gpd.GeoDataFrame(tr, geometry="geometry", crs=5186).reset_index(drop=True)
    j = gpd.sjoin(b[["bid", "geometry"]], tr[["F_SHIM", "label_kind", "date", "time", "geometry"]],
                  predicate="intersects")
    if j.empty:
        return None
    import shapely
    inter = shapely.area(shapely.intersection(j.geometry.values, tr.geometry.values[j.index_right.to_numpy()]))
    j = j.assign(inter=np.asarray(inter))
    g = j.groupby("bid").agg(inter=("inter", "sum"), depth=("F_SHIM", "max"),
                             survey=("label_kind", lambda s: (s == "조사").any()), date=("date", "min"),
                             time=("time", "min"))
    g["ratio"] = (g.inter / area[g.index]).clip(0, 1)
    return g[g.ratio >= min_ratio]


def labels(min_ratio=0.1):
    """flood=1: 사건에 붙은 흔적과 겹친 건물, flood=-1: 늦은 신고와만 겹친 건물(평가 제외)."""
    b = gpd.read_parquet(OUT / "buildings.parquet")
    area = b.geometry.area.to_numpy()
    ev = load_events()
    traces = {y[-4:]: load_trace(y[-4:]) for y in sorted(glob.glob(str(TRACE / "*")))}
    allt = pd.concat(traces.values(), ignore_index=True)
    parts = []
    for _, e in ev.iterrows():
        pos_days = str(e.trace_dates).split() if isinstance(e.trace_dates, str) else []
        late_days = str(e.late_dates).split() if isinstance(e.late_dates, str) else []
        if not pos_days and not late_days:
            continue
        g = _match(b, area, allt[allt.date.isin(pos_days)], min_ratio) if pos_days else None
        pos = g.reset_index().assign(flood=1) if g is not None else pd.DataFrame()
        gl = _match(b, area, allt[allt.date.isin(late_days)], min_ratio) if late_days else None
        exc = pd.DataFrame()
        if gl is not None:
            gl = gl[~gl.index.isin(pos.bid if len(pos) else [])]
            exc = gl.reset_index().assign(flood=-1)
        part = pd.concat([pos, exc], ignore_index=True)
        if len(part):
            part["event"] = e.event
            parts.append(part[["bid", "event", "flood", "depth", "survey", "date", "time", "ratio"]])
        log.info("%s 침수 반지하 후보 %d동, 늦은 신고로 제외 %d동", e.event, len(pos), len(exc))
    lab = pd.concat(parts, ignore_index=True)
    lab.to_parquet(OUT / "labels.parquet")
    log.info("라벨: 침수 %d, 제외 %d, 사건 %d개", int((lab.flood == 1).sum()), int((lab.flood == -1).sum()),
             lab.event.nunique())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["daily", "events", "rain10", "sewer_stations", "sewer", "buildings",
                                                 "labels"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.steps:
        {"daily": daily, "events": events, "rain10": rain10, "sewer_stations": sewer_stations,
         "sewer": sewer_history, "buildings": buildings, "labels": labels}[s]()
