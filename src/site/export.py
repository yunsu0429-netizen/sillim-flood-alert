r"""공개 웹(Vercel 정적 호스팅)용 JSON 내보내기.

정적 사이트는 서버가 없으므로 FastAPI 응답을 미리 파일로 만든다.
  static : 설정·건물·지형 타일·2022-08-08 재생 전 단계(상태·등급·건물 상세)  → site/data/
  live   : 지금 시각의 실시간 판정 1단계                                     → <out>/live/
           GitHub Actions가 10분마다 실행해 live-data 브랜치에 올리고, 화면은 raw.githubusercontent.com에서 읽는다.

개인정보 보호: 구독자 이메일·알림 기록이 섞이지 않도록 항상 빈 임시 DB(DB_PATH)로 실행하고,
개구부는 저장소의 data/manual/openings.csv만 넣는다. 건물 상세의 도로명 주소(네이버 키 필요)는 넣지 않는다.

실행: .venv\\Scripts\\python -m src.site.export static site/data
      .venv\\Scripts\\python -m src.site.export live out/
"""
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ["DB_PATH"] = os.environ.get("EXPORT_DB_PATH") or str(Path(tempfile.mkdtemp()) / "export.db")
os.environ["GMAIL_USER"] = ""            # 내보내기 중 메일 발송 금지
os.environ["GMAIL_APP_PASSWORD"] = ""
os.environ["NAVER_MAP_CLIENT_SECRET"] = ""   # 도로명 역지오코딩 호출 안 함

import numpy as np  # noqa: E402

log = logging.getLogger("export")
CHUNK = 200                              # 건물 상세 파일 하나에 담는 건물 수 (bid // CHUNK)


def _n(v, nd=2):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else round(v, nd)


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def _clean(o):
    """NaN → None (JSON 표준)."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return _clean(o.item())
    return o


def _boot(with_replay: bool):
    from src.config import settings
    assert settings.db_path == os.environ["DB_PATH"], "로컬 구독자 DB를 쓰면 안 됨"
    from src.alerts import subscriptions
    from src.api import main
    from src.api.runtime import LiveRuntime, ReplayRuntime
    from src.config import load_params
    subscriptions.init()
    params = load_params()
    main.live = LiveRuntime(params)
    main._seed_openings()
    main.live.engine.set_openings(subscriptions.all_openings())
    main.replay_rt = ReplayRuntime(params, main.live.engine.static) if with_replay else None
    return main


def building_static(rt) -> dict:
    """반지하 후보별 고정 정보 (건물 상세 패널용)."""
    from src.alerts import subscriptions
    st, m, e = rt.engine.static, rt.engine.model, rt.engine
    ops = {}
    for o in subscriptions.all_openings():
        ops.setdefault(o["bid"], []).append({k: o.get(k) for k in ("kind", "sill_above_ground_m", "barrier_m", "source")})
    out = {}
    for b in np.flatnonzero(st.is_banjiha.to_numpy()):
        r = st.iloc[b]
        j = int(e.st_idx[b])
        out[int(b)] = {"gid": str(r.gid) if "gid" in st.columns else None, "use": r.use_name, "structure": r.structure,
                       "approved": r.approved, "fu": int(r.floors_up), "fd": int(r.floors_down), "gl": _n(r.gl_m),
                       "rel": _n(r.rel_elev_m, 1), "slope": _n(r.slope_pct, 1), "sill": _n(m.sill[b]),
                       "barrier": _n(m.barrier[b]), "sq": m.op_quality[b], "st": j, "sd": _n(e.st_dist[b], 0) if j >= 0 else None,
                       "dl": _n(m.delta[b], 1), "src": getattr(r, "src_label", None), "ops": ops.get(int(b), [])}
    return out


def step_arrays(rt, snap) -> dict:
    """한 단계의 건물별 수치 (반지하 후보만): 시점별 등급, 점수, 물깊이·개구부 초과(지금·60분), 이용률, 요소 점수."""
    r = rt.engine.model.compute(snap.ctx)
    bj = np.flatnonzero(rt.engine.banjiha)
    out = {}
    for b in bj:
        out[int(b)] = {"g": int(snap.grades[b]), "l": [int(v) for v in snap.levels[:, b]], "s": _n(snap.scores[b], 1),
                       "d": [_n(r["depth"][0, b], 3), _n(r["depth"][-1, b], 3)],
                       "x": [_n(r["opening_excess"][0, b], 3), _n(r["opening_excess"][-1, b], 3)],
                       "u": _n(r["util"][0, b], 2), "c": [_n(r[f"s_{k}"][0, b], 0) for k in ("opening", "sewer", "rain", "terrain")],
                       "tf": int(snap.t_flood[b]), "cf": int(snap.confidence[b])}
    return out


def state_extra(rt, snap) -> dict:
    g = snap.ctx.gate
    rf = snap.ctx.rain[0] or {}
    return {"gate": None if g is None else [bool(v) for v in g], "rain_i30": _n(rf.get("I30"), 1),
            "theta_shift": (rt.p["risk_model"].get("district_gate") or {}).get("theta_shift", 0),
            "level_high": rt.p["risk_model"]["levels"][1]}


def export_static(out: Path):
    from src.config import PROCESSED
    from src.api.terrain import terrain_tile
    main = _boot(with_replay=True)
    live, rp = main.live, main.replay_rt
    cfg = _clean(main.config())
    cfg.update({"static": True, "gmail": False, "naver_client_id": os.environ.get("NAVER_MAP_CLIENT_ID") or None,
                "live_base": os.environ.get("LIVE_BASE", "https://raw.githubusercontent.com/yunsu0429-netizen/sillim-flood-alert/live-data/live/"), "chunk": CHUNK,
                "streamlit_url": os.environ.get("STREAMLIT_URL", "https://sillim-flood-alert.streamlit.app")})
    cfg["modes"]["live"]["label"] = "실시간: 서울시 강우량계 신림P·관악구청 + 기상청 초단기예보 (10분마다 갱신)"
    _dump(out / "config.json", cfg)
    (out / "buildings.json").write_bytes((PROCESSED / "buildings_3d.json").read_bytes())
    st = live.engine.static
    _dump(out / "gid.json", {str(g): int(b) for g, b in zip(st.gid.astype(str), st.bid)} if "gid" in st.columns else {})
    # 재생 전 단계
    bs = building_static(rp)
    steps = []
    for k, snap in enumerate(rp.snaps):
        stt = _clean(main.state("replay", k)) | state_extra(rp, snap)
        _dump(out / "replay" / f"state_{k}.json", stt)
        _dump(out / "replay" / f"risk_{k}.json", _clean(main.risk("replay", k)))
        steps.append(step_arrays(rp, snap))
        log.info("재생 %d/%d", k + 1, len(rp.snaps))
    from src.alerts import detector
    chunks: dict[int, dict] = {}
    for b, info in bs.items():
        chunks.setdefault(b // CHUNK, {})[b] = {"static": info, "steps": [s[b] for s in steps],
                                                "timeline": detector.timeline(rp.snaps, b, rp.p)}
    for c, d in chunks.items():
        _dump(out / "replay" / "detail" / f"{c}.json", _clean(d))
    # 지형 타일 (줌 12~16, 신림동 범위)
    w, s_, e, n = json.loads((PROCESSED / "buildings_3d.json").read_text(encoding="utf-8"))["meta"]["bounds"]
    pad = 0.012
    cnt = 0
    for z in range(12, 17):
        x0, y1 = _tile(w - pad, s_ - pad, z)
        x1, y0 = _tile(e + pad, n + pad, z)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                p = out / "terrain" / str(z) / str(x) / f"{y}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(terrain_tile(z, x, y))
                cnt += 1
    log.info("정적 내보내기 완료: 재생 %d단계, 상세 %d묶음, 지형 타일 %d장", len(rp.snaps), len(chunks), cnt)


def _tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def export_live(out: Path):
    from datetime import datetime, timedelta
    from src.sources import sewer_api
    try:                                   # SWMM 6시간 적분용 하수관 수위 이력 (서버처럼 누적된 DB가 없으므로)
        now = datetime.now()
        sewer_api.store(sewer_api.fetch(now - timedelta(hours=3), now))
    except Exception as e:
        log.warning("하수관 이력 수집 실패: %s", e)
    main = _boot(with_replay=False)
    live = main.live
    live._fetch_live()
    live._run_cycle(datetime.now())
    snap = live.snap
    stt = _clean(main.state("live", 0)) | state_extra(live, snap)
    stt["alerts"] = []                                   # 알림 기록은 공개하지 않음
    stt["exported_at"] = datetime.now().isoformat(timespec="seconds")
    stt["version"] = f"live-{stt['cycle_time']}"          # 실행마다 새 프로세스라 판정 시각으로 버전 구분
    rk = _clean(main.risk("live", 0))
    rk["version"] = stt["version"]
    _dump(out / "state.json", stt)
    _dump(out / "risk.json", rk)
    bs = building_static(live)
    arr = step_arrays(live, snap)
    chunks: dict[int, dict] = {}
    for b, info in bs.items():
        chunks.setdefault(b // CHUNK, {})[b] = {"static": info, "steps": [arr[b]], "timeline": None}
    for c, d in chunks.items():
        _dump(out / "detail" / f"{c}.json", _clean(d))
    log.info("실시간 내보내기 완료 %s: %s", stt["cycle_time"], stt["counts"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # 요청 URL에 인증키가 들어 있어 공개 로그에 남기지 않음
    what, dest = sys.argv[1], Path(sys.argv[2])
    {"static": export_static, "live": export_live}[what](dest)
