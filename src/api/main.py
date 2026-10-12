"""FastAPI 서버. 실행: .venv\\Scripts\\python -m uvicorn src.api.main:app --port 5173"""
import asyncio
import csv
import logging
import math
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import numpy as np
import pyproj
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from shapely.geometry import Point

from src.alerts import detector, mailer, subscriptions
from src.api.runtime import LiveRuntime, ReplayRuntime
from src.api.terrain import terrain_tile
from src.config import MANUAL, PROCESSED, ROOT, load_params, settings
from src.engine.rules import GRADES, GRADES_KO
from src.geo import naver
from src.sources import seoul_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
live: LiveRuntime | None = None
replay_rt: ReplayRuntime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global live, replay_rt
    subscriptions.init()
    params = load_params()
    live = LiveRuntime(params)
    replay_rt = ReplayRuntime(params, live.engine.static)
    st = live.engine.static
    if "gid" in st.columns:   # 건물 목록이 갱신됐어도 구독·개구부가 같은 건물을 가리키게 재연결
        stats = subscriptions.relink(st["gid"].astype(str).tolist())
        log.info("구독·개구부 건물 재연결: %s", stats)
        if stats["missing"]:
            log.warning("건물 목록에서 사라진 건물의 구독·개구부 %d건 — 화면에 표시되지 않음", stats["missing"])
    else:
        log.warning("buildings_static.parquet에 gid 없음 → build_static을 다시 실행해야 갱신 시 구독이 보호됨")
    if _seed_openings():
        _refresh_openings()
    if not mailer.configured():
        log.warning("GMAIL_USER / GMAIL_APP_PASSWORD 미설정 → 메일은 data/outbox/ 에 .eml로만 저장됩니다")
    if not settings.admin_token:
        log.warning("ADMIN_TOKEN 미설정 → 개구부 입력이 누구에게나 열려 있습니다 (공개 배포 시 설정 필요)")
    task = asyncio.create_task(live.loop())
    yield
    task.cancel()


def _seed_openings():
    """data/manual/openings.csv의 수동 개구부를 DB에 한 번만 넣는다 (건물에 개구부가 없을 때만)."""
    path = MANUAL / "openings.csv"
    if not path.exists():
        return
    st = live.engine.static
    seeded = set()
    n = 0
    for row in csv.DictReader(open(path, encoding="utf-8")):
        hit = st[st.jibun == row["jibun"]]
        if not len(hit):
            log.warning("openings.csv 지번 %s 없음", row["jibun"])
            continue
        bid = int(hit.iloc[0].bid)
        if bid not in seeded and subscriptions.openings_for(bid):
            continue
        seeded.add(bid)
        num = lambda k: float(row[k]) if row.get(k) else None
        subscriptions.add_opening(bid, {"kind": row["kind"], "sill_above_ground_m": num("sill_above_ground_m"),
                                        "width_m": num("width_m"), "height_m": num("height_m"),
                                        "barrier_m": num("barrier_m"), "source": row["source"], "note": row["note"],
                                        "quality": row.get("quality") or "manual"})
        n += 1
    return n


app = FastAPI(title="신림동 반지하 침수 위험", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
_to5186 = pyproj.Transformer.from_crs(4326, 5186, always_xy=True)


def _num(v, nd=1):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


def _view(mode: str, step: int):
    """(런타임, 스냅샷, 버전 문자열). 재생은 브라우저가 고른 단계(step)를 그대로 보여준다."""
    if mode == "live":
        return live, live.snap, f"live-{live.version}"
    if mode == "replay":
        step, snap = replay_rt.snap_at(step)
        return replay_rt, snap, f"replay-{step}"
    raise HTTPException(400, f"알 수 없는 모드: {mode}")


# ---------- 간단한 요청 제한 (공개 배포 시 메일 발송 남용 방지) ----------
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _limit(key: str, n: int, per_sec: int):
    now = time.time()
    with _hits_lock:
        q = _hits[key]
        while q and now - q[0] > per_sec:
            q.popleft()
        if len(q) >= n:
            raise HTTPException(429, "요청이 너무 많습니다. 잠시 후 다시 시도하세요.")
        q.append(now)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")  # 클라우드 프록시 뒤에서는 실제 IP가 여기에
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


@app.get("/api/health")
def health():
    return {"ok": True, "live_version": live.version}


@app.get("/api/config")
def config():
    return {"default_mode": settings.mode if settings.mode in ("live", "replay") else "live",
            "modes": {"live": {"label": live.data_label}, "replay": {"label": replay_rt.data_label}},
            "replay_times": [t.strftime("%H:%M") for t in replay_rt.times],
            "naver_client_id": settings.naver_map_client_id or None,
            "thresholds": live.p["score"]["thresholds"], "grades": GRADES, "grades_ko": GRADES_KO,
            "gmail": mailer.configured(), "openings_locked": bool(settings.admin_token),
            "notice": "위험도는 가정 가중치로 계산한 참고 지표이며 침수 확률이 아닙니다. 공식 재난문자를 우선하세요."}


@app.get("/api/buildings")
def buildings():
    return FileResponse(PROCESSED / "buildings_3d.json", media_type="application/json",
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/terrain/{z}/{x}/{y}.png")
def terrain(z: int, x: int, y: int):
    if not 10 <= z <= 18:
        raise HTTPException(404)
    return Response(terrain_tile(z, x, y), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


def _sewer_summary(rt, s):
    """수위계별 최신 수위·상승속도·이용률(현재) — 지도 마커와 패널용."""
    out = []
    for j, st in enumerate(rt.engine.stations):
        f = s.ctx.station_feat[j]
        util = s.ctx.util[0][j] if f is not None else None
        out.append({"station": st["station"], "lng": st["lng"], "lat": st["lat"], "query": st.get("query"),
                    "level_m": None if f is None else f["H0"], "last_ts": None if f is None else _iso(f["last_ts"]),
                    "rise30_cm_min": None if f is None or f["rise30"] is None else round(f["rise30"] * 100, 2),
                    "status": None if f is None else f["status"], "h_crit": round(float(rt.engine.h_crit[j]), 2),
                    "h_crit_src": rt.engine.h_crit_src[j], "util": _num(util, 2)})
    return out


def _iso(t):
    return t.isoformat(timespec="minutes") if t else None


@app.get("/api/state")
def state(mode: str = "live", step: int = 0):
    rt, s, version = _view(mode, step)
    counts = rt.engine.counts(s.grades)
    order = np.argsort(-np.nan_to_num(s.scores, nan=-1))[:20]
    top = [{"bid": int(b), "jibun": rt.engine.static.iloc[b].jibun, "score": _num(s.scores[b], 0),
            "grade": int(s.grades[b]), "t_flood": int(s.t_flood[b])} for b in order if not np.isnan(s.scores[b])]
    hz = s.ctx.horizons
    fc = {h: {"counts": rt.engine.counts(s.levels[k].astype(int)),
              "rain60": _num((s.ctx.rain[h] or {}).get("P60"))} for k, h in enumerate(hz) if h in (30, 60)}
    rf = s.rain_now
    conf = s.confidence[rt.engine.banjiha]
    cl = rt.p["risk_model"]["confidence"]["levels"]
    out = {"mode": mode, "version": version, "sim_time": s.t.isoformat(timespec="seconds"),
           "cycle_time": s.t.isoformat(timespec="minutes"), "model_version": rt.p["risk_model"]["model_version"],
           "rain": {"times": [t.strftime("%H:%M") for t, _ in rt.series], "mm": [v for _, v in rt.series]},
           "current": None if rf is None else {"r60": _num(rf["P60"]), "i10": _num(rf["I10"]), "i30": _num(rf["I30"]),
                                               "p180": _num(rf["P180"])},
           "counts": counts, "top": top, "forecast": fc, "alerts": [], "forecast_rain": None, "live": None,
           "flood_soon": int(((s.t_flood > 0) & rt.engine.banjiha).sum()),
           "confidence": {"HIGH": int((conf >= cl[1]).sum()), "MEDIUM": int(((conf >= cl[0]) & (conf < cl[1])).sum()),
                          "LOW": int((conf < cl[0]).sum())},
           "openings": {"buildings_with_input": rt.engine.model.imputed["n"] if rt.engine.model.imputed else 0,
                        "imputed_sill_m": _num(rt.engine.model.imputed["sill"], 2) if rt.engine.model.imputed else None},
           "sewer": _sewer_summary(rt, s),
           "vulnerable": [int(b) for b in np.flatnonzero(rt.engine.vulnerable)],
           "vulnerable_top_pct": (rt.p["risk_model"].get("vulnerable") or {}).get("top_pct", 10)}
    if mode == "live":
        last_obs = live.series[-1][0] if live.series else None
        out["sim_time"] = datetime.now().isoformat(timespec="seconds")
        out["alerts"] = subscriptions.recent_alerts(8, (datetime.now() - timedelta(days=1)).isoformat(timespec="minutes"))
        out["forecast_rain"] = {**live.forecast_info,
                                "times": [t.strftime("%H:%M") for t, _ in live.forecast_series if not last_obs or t > last_obs],
                                "mm": [round(v, 2) for t, v in live.forecast_series if not last_obs or t > last_obs]}
        out["live"] = {**live.rain_info, "error": live.fetch_error, "sewer_error": live.sewer_error,
                       "last_fetch": live.last_fetch.isoformat(timespec="seconds"),
                       "min_steps": seoul_api.MIN_STEPS_R60,
                       "steps_last_hour": seoul_api.steps_in_last_hour(live.series, s.t)}
    else:
        out["step"] = replay_rt.snap_at(step)[0]
    return out


@app.get("/api/risk")
def risk(mode: str = "live", step: int = 0):
    _, s, version = _view(mode, step)
    return {"version": version, "grades": s.grades.tolist(),
            "scores": [None if math.isnan(v) else round(float(v)) for v in s.scores],
            "t_flood": s.t_flood.tolist()}


@app.get("/api/building/{bid}")
def building(bid: int, mode: str = "live", step: int = 0):
    rt, s, _ = _view(mode, step)
    if not 0 <= bid < len(rt.engine.static):
        raise HTTPException(404, "건물 없음")
    r = rt.engine.static.iloc[bid]
    base = {"bid": bid, "gid": str(r.gid) if "gid" in rt.engine.static.columns else None,
            "address": rt.address_of(bid), "road": rt.road_address(bid),
            "use": r.use_name, "structure": r.structure,
            "approved": r.approved, "floors_up": int(r.floors_up), "floors_down": int(r.floors_down),
            "height_m": _num(r.height_m), "is_banjiha": bool(r.is_banjiha), "gl_m": _num(r.gl_m, 2),
            "rel_elev_m": _num(r.rel_elev_m), "slope_pct": _num(r.slope_pct),
            "openings": subscriptions.openings_for(bid), "subscribers": subscriptions.subscriber_count(bid),
            "source": getattr(r, "src_label", None),
            "gl_note": "지반고는 90m 해상도 DEM 보간값으로 실제와 수 m 차이가 날 수 있음",
            "vulnerable": bool(rt.engine.vulnerable[bid])}
    if not r.is_banjiha:
        return base | {"grade": -1, "reasons": ["반지하 후보가 아님 (지하층 없음 또는 비주거)"]}
    d = rt.engine.detail(s, bid)
    m = rt.engine.model
    cl = rt.p["risk_model"]["confidence"]["levels"]
    st = rt.engine.station_of(bid)
    sewer = None
    if st:
        j, sinfo, dist = st
        f = s.ctx.station_feat[j]
        sewer = {"station": sinfo["station"], "location": sinfo.get("query"), "distance_m": round(dist),
                 "level_m": None if f is None else f["H0"],
                 "rise30_cm_min": None if f is None or f["rise30"] is None else round(f["rise30"] * 100, 2),
                 "h_crit_m": round(float(rt.engine.h_crit[j]), 2), "h_crit_src": rt.engine.h_crit_src[j],
                 "util_now": d["util"][0]}
    oq = m.op_quality[bid]
    t_eta = (s.t + timedelta(minutes=d["t_flood"])).strftime("%H:%M") if d["t_flood"] > 0 else None
    return base | {
        "model_version": rt.p["risk_model"]["model_version"],
        "index_label": "침수위험지수 v0.3 (서울 침수흔적도 보정 + 자치구 침수예보 결합, 참고 지표)",
        "score": _num(d["score"][0], 1), "grade": int(s.grades[bid]), "raw_level": d["level"][0],
        "forecast": {f"{h}min": {"level": lv, "score": _num(sc, 1)}
                     for h, lv, sc in zip(d["horizons"], d["level"], d["score"]) if h > 0},
        "horizons": d["horizons"], "levels": d["level"],
        "estimated_flood_time_min": d["t_flood"] if d["t_flood"] >= 0 else None, "estimated_flood_clock": t_eta,
        "estimated_protected_flood_time_min": d["t_protected"] if d["t_protected"] >= 0 else None,
        "prediction": {"water_surface_elevation_m": None if r.gl_m != r.gl_m or d["depth"][0] is None
                       else round(float(r.gl_m) + d["depth"][0], 3),
                       "ground_flood_depth_m": d["depth"], "opening_excess_depth_m": d["opening_excess"],
                       "protected_excess_depth_m": d["protected_excess"], "sewer_utilization": d["util"]},
        "components": d["components"],
        "opening": {"sill_m": _num(m.sill[bid], 2), "barrier_m": _num(m.barrier[bid], 2), "quality": oq,
                    "imputed_from": m.imputed["n"] if oq == "imputed" and m.imputed else None},
        "sewer": sewer,
        "confidence": {"score": d["confidence"],
                       "level": "HIGH" if d["confidence"] >= cl[1] else "MEDIUM" if d["confidence"] >= cl[0] else "LOW"},
        "reasons": rt.engine.reasons(s, bid),
        "alert_timeline": detector.timeline(replay_rt.snaps[:replay_rt.snap_at(step)[0] + 1], bid, rt.p)
        if mode == "replay" else None}


@app.get("/api/lookup")
def lookup(gid: str):
    """건물 고유 ID → 현재 bid. 링크는 gid로 만들어 건물 목록을 갱신해도 같은 건물을 가리키게 한다."""
    st = live.engine.static
    hit = st.index[st["gid"].astype(str) == gid] if "gid" in st.columns else []
    if not len(hit):
        raise HTTPException(404, "건물 목록에 없는 건물입니다 (데이터 갱신으로 사라졌을 수 있음)")
    return {"bid": int(hit[0])}


def _bid_at(lng: float, lat: float):
    x, y = _to5186.transform(lng, lat)
    pt = Point(x, y)
    st = live.engine.static
    hit = st[st.geometry.contains(pt)]
    if len(hit):
        return int(hit.iloc[0].bid)
    d = st.geometry.distance(pt)
    return int(st.iloc[int(d.argmin())].bid) if d.min() < 30 else None


@app.get("/api/search")
def search(q: str, request: Request):
    q = q.strip()[:100]
    st = live.engine.static
    jibun = q.split("신림동")[-1].strip() if "신림동" in q else q
    exact = st[st.jibun == jibun]
    if len(exact):
        return {"source": "지번 검색", "results": [{"bid": int(b), "address": live.address_of(int(b))} for b in exact.bid]}
    if naver.enabled():
        _limit(f"geo:{_client_ip(request)}", 30, 60)  # 네이버 유료 호출 보호
        try:
            hits = naver.geocode(q if "관악구" in q else f"서울특별시 관악구 {q}")
        except Exception as e:
            return {"source": "네이버 Geocoding 실패", "error": str(e), "results": []}
        res = []
        for h in hits:
            bid = _bid_at(h["lng"], h["lat"])
            if bid is not None:
                res.append({"bid": bid, "address": h["road"] or h["address"]})
        return {"source": "네이버 Geocoding", "results": res}
    part = st[st.jibun.str.startswith(jibun)].head(10)
    return {"source": "지번 앞부분 검색", "results": [{"bid": int(b), "address": live.address_of(int(b))} for b in part.bid]}


class SubReq(BaseModel):
    bid: int
    email: str


@app.post("/api/subscribe")
def subscribe(req: SubReq, request: Request):
    if not 0 <= req.bid < len(live.engine.static):
        raise HTTPException(404, "건물 없음")
    email = req.email.strip().lower()[:200]
    _limit(f"sub-ip:{_client_ip(request)}", 5, 3600)   # 같은 IP 시간당 5건
    _limit(f"sub-mail:{email}", 3, 86400)              # 같은 주소로 하루 3번까지 확인 메일
    try:
        res = subscriptions.subscribe(req.bid, email)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if res["status"] == "already_active":
        return {"status": "already_active", "message": "이미 알림이 등록된 이메일입니다."}
    channel = detector.confirmation_mail(email, res["token"], live.address_full(req.bid))
    msg = "확인 메일을 보냈습니다. 메일의 링크를 눌러야 알림이 시작됩니다."
    if channel == "outbox":
        msg = "메일 발송이 아직 설정되지 않아 확인 메일을 보내지 못했습니다 (서버에만 저장됨)."
    return {"status": "pending", "message": msg}


def _page(title, body):
    return HTMLResponse(f"<!doctype html><meta charset=utf-8><title>{title}</title>"
                        f"<body style='font-family:sans-serif;max-width:520px;margin:60px auto'>"
                        f"<h2>{title}</h2><p>{body}</p><p><a href='/'>대시보드로 이동</a></p>")


@app.get("/api/confirm")
def confirm(token: str):
    row = subscriptions.confirm(token)
    if not row:
        return _page("링크가 유효하지 않습니다", "이미 수신 거부했거나 잘못된 링크입니다.")
    return _page("알림이 시작되었습니다", f"{live.address_of(row['bid'])}의 위험 등급이 올라가면 메일로 알려드립니다.")


@app.get("/api/unsubscribe")
def unsubscribe(token: str):
    ok = subscriptions.unsubscribe(token)
    return _page("수신 거부 완료" if ok else "이미 처리된 링크", "이메일 정보를 삭제했습니다." if ok else "")


class OpeningReq(BaseModel):
    kind: str
    sill_above_ground_m: float
    width_m: float | None = None
    height_m: float | None = None
    barrier_m: float | None = None
    note: str | None = None
    quality: str | None = None


@app.post("/api/building/{bid}/openings")
def add_opening(bid: int, req: OpeningReq, x_admin_token: str | None = Header(default=None)):
    if settings.admin_token and not secrets.compare_digest(x_admin_token or "", settings.admin_token):
        raise HTTPException(403, "개구부 입력은 관리자만 할 수 있습니다 (관리자 링크로 접속하세요)")
    if not 0 <= bid < len(live.engine.static):
        raise HTTPException(404, "건물 없음")
    if req.kind not in ("window", "door") or not -3 <= req.sill_above_ground_m <= 5:
        raise HTTPException(400, "종류는 창문/출입구, 문턱 높이는 -3 ~ 5 m 범위로 입력하세요")
    quality = req.quality if req.quality in ("manual", "ai_estimate", "observed") else "manual"
    src = {"ai_estimate": "Claude AI 육안 추정(네이버 거리뷰)", "observed": "현장 실측"}.get(quality, "네이버 거리뷰 육안 판독")
    data = req.model_dump() | {"note": (req.note or "")[:200], "source": src, "quality": quality}
    oid = subscriptions.add_opening(bid, data)
    _refresh_openings()
    return {"id": oid}


def _refresh_openings():
    """개구부가 바뀌면 평균 대체값이 달라지므로 실시간은 즉시, 재생은 백그라운드로 다시 계산."""
    live.refresh_openings()
    replay_rt.engine.set_openings(subscriptions.all_openings())
    threading.Thread(target=replay_rt.recompute, daemon=True).start()


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
