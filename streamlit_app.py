"""Streamlit Community Cloud용 신림동 반지하 침수 위험 대시보드 (판정 엔진을 직접 실행).

- 실시간: 서울시 강우량계·하수관 수위·기상청 초단기예보를 받아 10분마다 다시 판정 (앱이 깨어 있는 동안)
- 2022-08-08 사례: 실측 강우 재생
인증키는 Streamlit 화면의 Secrets(SEOUL_OPEN_API_KEY, KMA_API_KEY)에만 넣는다. 저장소에는 키가 없다.
공개 앱이라 이메일 알림 신청·개구부 입력은 받지 않는다 (개인정보를 저장하지 않음).
"""
import os
import tempfile
import time
from pathlib import Path

# 서울시·기상청 자료는 한국 시간(시간대 표기 없음) → 서버(UTC) 시간대를 맞춘다
os.environ["TZ"] = "Asia/Seoul"
if hasattr(time, "tzset"):
    time.tzset()

import streamlit as st  # noqa: E402

KEYS = ("SEOUL_OPEN_API_KEY", "KMA_API_KEY")
for k in KEYS:
    try:
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])   # Secrets를 바꾸면 다음 실행 때 바로 반영
    except Exception:          # secrets.toml이 없으면 (로컬 실행) .env 사용
        pass
os.environ["DB_PATH"] = str(Path(tempfile.gettempdir()) / "sillim_streamlit.db")   # 구독자 없는 빈 DB
os.environ["GMAIL_USER"] = ""
os.environ["GMAIL_APP_PASSWORD"] = ""
os.environ["NAVER_MAP_CLIENT_SECRET"] = ""

import logging  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)   # 요청 URL에 인증키가 들어 있어 로그에 남기지 않음

import json  # noqa: E402
import threading  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402

from src.config import PROCESSED, load_params, settings  # noqa: E402

for k in KEYS:                 # 이미 불러온 설정 객체에도 반영 (앱 재시작 없이 키 변경 적용)
    if os.environ.get(k):
        setattr(settings, k.lower(), os.environ[k])
KEY_SIG = tuple(bool(os.environ.get(k)) for k in KEYS)

GRADE_KO = ["낮음", "주의", "경고", "위험"]
COLORS = [[46, 158, 91], [242, 194, 48], [240, 124, 30], [214, 49, 43]]
NONE = [154, 154, 154]
VERCEL_URL = os.environ.get("VERCEL_URL_PUBLIC", "")
try:
    VERCEL_URL = VERCEL_URL or str(st.secrets.get("VERCEL_URL_PUBLIC", ""))
except Exception:
    pass

st.set_page_config(page_title="신림동 반지하 침수 위험", page_icon="🌧️", layout="wide")


# ---------------------------------------------------------------- 엔진 (프로세스당 한 번)
@st.cache_resource(show_spinner="판정 엔진 준비 중 (처음 한 번, 약 1분)…")
def runtimes(key_sig):
    from src.alerts import subscriptions
    from src.api import main
    from src.api.runtime import LiveRuntime, ReplayRuntime
    subscriptions.init()
    params = load_params()
    live = LiveRuntime(params)
    main.live = live
    main._seed_openings()
    live.engine.set_openings(subscriptions.all_openings())
    replay = ReplayRuntime(params, live.engine.static)
    return live, replay, threading.Lock()


@st.cache_resource
def buildings():
    return json.loads((PROCESSED / "buildings_3d.json").read_text(encoding="utf-8"))


def live_refresh(live, lock, max_age_min=10):
    """마지막 판정이 10분 넘었으면 다시 수집·판정 (여러 사람이 동시에 열어도 한 번만)."""
    with lock:
        if live.snap is None or (datetime.now() - live.snap.t).total_seconds() > max_age_min * 60:
            live._fetch_live()
            live._fetch_forecast()
            live._run_cycle(datetime.now())
    return live.snap


# ---------------------------------------------------------------- 화면
live, replay, lock = runtimes(KEY_SIG)     # 키가 새로 생기면 엔진을 다시 만든다
B = buildings()
st.title("신림동 반지하 침수 위험")
st.caption("관악구 신림동 반지하 후보 11,523동 · 침수위험지수 v0.3 (서울 침수흔적도 보정 + 자치구 침수예보 결합). "
           "참고 지표이며 침수 확률이 아닙니다. 공식 재난문자를 우선하세요.")
if VERCEL_URL:
    st.markdown(f"3D 지도 웹 버전: [{VERCEL_URL}]({VERCEL_URL})")

mode = st.sidebar.radio("화면", ["실시간 강우", "2022-08-08 사례"], index=0)
if mode == "실시간 강우":
    rt, snap = live, live_refresh(live, lock)
    st.sidebar.write(f"판정 {snap.t:%m/%d %H:%M} · 10분마다 갱신" + ("" if all(KEY_SIG) else " · 인증키 없음(샘플 자료)"))
    if st.sidebar.button("지금 다시 판정"):
        with lock:
            live._fetch_live(); live._fetch_forecast(); live._run_cycle(datetime.now())
        st.rerun()
else:
    rt = replay
    times = [t.strftime("%H:%M") for t in replay.times]
    pick = st.sidebar.select_slider("재생 시각 (2022-08-08~09)", options=times, value="20:40")
    snap = replay.snaps[times.index(pick)]

eng = rt.engine
rf = snap.ctx.rain[0]
counts = eng.counts(snap.grades)
c = st.columns(6)
c[0].metric("판정 시각", f"{snap.t:%H:%M}")
c[1].metric("60분 강우", "-" if rf is None else f"{rf['P60']:.0f} mm")
for i, k in enumerate(["주의", "경고", "위험"]):
    c[2 + i].metric(k, f"{counts[k]:,}동")
g = snap.ctx.gate
c[5].metric("관악구 기준", "자료 없음" if g is None else ("충족" if g[0] else "미충족"))

# 3D 지도
grades = snap.grades
rows = []
for b in B["buildings"]:
    gr = int(grades[b["id"]]) if b["b"] else -2
    col = (COLORS[gr] if gr >= 0 else NONE) + [220] if b["b"] else [180, 182, 186, 90]
    rows.append({"polygon": b["p"], "h": b["h"] + (b.get("bd") or 0), "color": col, "jibun": b["j"],
                 "grade": GRADE_KO[gr] if gr >= 0 else ("데이터 없음" if gr == -1 else "반지하 아님")})
w, s_, e, n = B["meta"]["bounds"]
layer = pdk.Layer("PolygonLayer", rows, get_polygon="polygon", get_elevation="h", get_fill_color="color",
                  extruded=True, pickable=True, auto_highlight=True)
view = pdk.ViewState(longitude=(w + e) / 2, latitude=(s_ + n) / 2, zoom=14, pitch=50, bearing=-20)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="light",
                         tooltip={"text": "신림동 {jibun}\n{grade}"}), height=560)
st.caption("색: 초록 낮음 · 노랑 주의 · 주황 경고 · 빨강 위험 · 회색 반지하 후보 아님. 경고 이상은 관악구 강우량계가 서울시 침수예보 기준을 넘었을 때만.")

left, right = st.columns([1, 1])
with left:
    st.subheader("위험 상위 건물")
    order = np.argsort(-np.nan_to_num(snap.scores, nan=-1))[:15]
    st.dataframe(pd.DataFrame([{"지번": eng.static.iloc[b].jibun, "점수": round(float(snap.scores[b])),
                                "등급": GRADE_KO[int(snap.grades[b])] if snap.grades[b] >= 0 else "-",
                                "개구부 유입": "유입 중" if snap.t_flood[b] == 0 else (f"{snap.t_flood[b]}분 뒤" if snap.t_flood[b] > 0 else "-")}
                               for b in order if not np.isnan(snap.scores[b])]), hide_index=True, width="stretch")
    st.subheader("강우 (10분, mm)")
    series = [(t, v) for t, v in rt.series if t <= snap.t][-36:]
    st.bar_chart(pd.DataFrame({"mm": [v for _, v in series]}, index=[t.strftime("%H:%M") for t, _ in series]))
with right:
    st.subheader("건물 조회")
    q = st.text_input("신림동 지번 (예: 419-1)", value="419-1").strip().replace("신림동", "").strip()
    hit = eng.static[eng.static.jibun == q]
    if not len(hit):
        st.info("해당 지번이 없습니다.")
    else:
        bid = int(hit.iloc[0].bid)
        r = eng.static.iloc[bid]
        if not r.is_banjiha:
            st.write("반지하 후보가 아닙니다 (지하층 없음 또는 비주거).")
        else:
            d = eng.detail(snap, bid)
            gr = int(snap.grades[bid])
            st.markdown(f"**서울특별시 관악구 신림동 {q}** · {r.use_name} · 지상 {int(r.floors_up)}층 지하 {int(r.floors_down)}층")
            sc = d["score"][0]
            st.markdown(f"### {GRADE_KO[gr] if gr >= 0 else '판정 불가'}" + (f" {sc:.0f}점" if sc is not None else ""))
            st.dataframe(pd.DataFrame({"시점": ["지금"] + [f"+{h}분" for h in d["horizons"][1:]],
                                       "등급": [GRADE_KO[v] if v >= 0 else "-" for v in d["level"]],
                                       "물깊이(cm)": [None if v is None else round(v * 100) for v in d["depth"]]}).T,
                         width="stretch")
            st.markdown("**판정 근거**")
            for line in eng.reasons(snap, bid):
                st.markdown(f"- {line}")
            if rt is replay:
                from src.alerts import detector
                k = replay.snaps.index(snap)
                tl = detector.timeline(replay.snaps[:k + 1], bid, replay.p)
                if tl:
                    st.markdown("**알림 타임라인 (재생, 실제 발송 없음)**")
                    for a in tl:
                        st.markdown(f"- {a['t'][11:16]} " + (f"사전경보: {a['minutes']}분 뒤 {GRADE_KO[a['level']]} 예상"
                                                             if a["kind"] == "pre" else f"현재 경보: {GRADE_KO[a['level']]}"))

st.divider()
st.caption("자료: 서울 열린데이터광장(강우량·하수관로 수위·침수흔적도·침수예상도), 기상청 초단기예보·AWS, 국토교통부 GIS건물통합정보, "
           "국토지리정보원 DEM · 배경지도 © CARTO, OpenStreetMap contributors")
