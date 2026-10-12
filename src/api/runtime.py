"""실시간 계산 루프와 2022-08-08 재생 데이터.

- LiveRuntime: 서울시 강우·하수관 수위(2분마다 확인)와 기상청 초단기예보(10분마다)로 10분 주기 판정, 알림 규칙에 따라 메일 발송.
- ReplayRuntime: 2022-08-08 재생 결과를 서버 시작 때 10분 단위로 전부 미리 계산해 둔다.
  재생 시각은 각 사용자의 브라우저가 고르므로 한 사람의 재생 조작이 다른 사람 화면에 영향을 주지 않고,
  재생 중에는 메일을 보내지 않는다. 하수관은 2022-08-08 실측(관악구 수위계 7곳), 강우는 가정 하이토그래프.
"""
import asyncio
import json
import logging
import threading
from datetime import datetime, timedelta

import numpy as np
import pyproj

from src.alerts import detector, subscriptions
from src.config import load_params
from src.engine.forecast import RiskEngine, Snapshot
from src.geo import naver
from src.sources import kma_api, replay, seoul_api, sewer_api

log = logging.getLogger(__name__)
FETCH_EVERY = timedelta(minutes=2)  # 강우 자료는 10분마다 갱신되지만 수신 지연을 줄이려고 2분마다 확인
FORECAST_EVERY = timedelta(minutes=10)  # 초단기예보는 30분마다 발표
_TO_4326 = pyproj.Transformer.from_crs(5186, 4326, always_xy=True)


class _Base:
    mode = ""

    def __init__(self, params: dict | None, static, stations: list):
        self.p = params or load_params()
        self.engine = RiskEngine(self.p, static, stations)
        self.engine.set_openings(subscriptions.all_openings())
        self.cycle = timedelta(minutes=self.p["engine"]["cycle_minutes"])

    def address_of(self, bid: int) -> str:
        return f"서울특별시 관악구 신림동 {self.engine.static.iloc[bid].jibun}"

    def road_address(self, bid: int) -> dict | None:
        """네이버 Reverse Geocoding으로 도로명 주소·건물명. 키가 없거나 실패하면 None (판정·화면은 그대로)."""
        if not naver.enabled():
            return None
        pt = self.engine.static.geometry.iloc[bid].representative_point()
        lng, lat = _TO_4326.transform(pt.x, pt.y)
        try:
            return naver.reverse(lng, lat)
        except Exception as e:
            log.warning("도로명 주소 조회 실패(bid=%s): %s", bid, e)
            return None

    def address_full(self, bid: int) -> str:
        """메일용: 지번 주소 + (도로명 주소 건물명)."""
        road = self.road_address(bid) or {}
        extra = " ".join(p for p in [road.get("road"), road.get("building")] if p)
        return f"{self.address_of(bid)} (도로명: {extra})" if extra else self.address_of(bid)


class ReplayRuntime(_Base):
    mode = "replay"

    def __init__(self, params: dict | None = None, static=None):
        self.sewer_obs = sewer_api.load_replay() if sewer_api.REPLAY_CSV.exists() else {}
        # 재생에서는 그날 자료가 있는 수위계만 연결 (나머지는 2022년에 없던 수위계)
        stations = [s for s in sewer_api.load_stations() if s["station"] in self.sewer_obs]
        super().__init__(params, static, stations)
        self.series = replay.load_series()
        self.gauges = replay.load_gauges()          # 관악구 서울시 강우량계 (자치구 침수예보 기준)
        self.data_label = replay.LABEL + f" · 하수관 2022-08-08 실측 {len(stations)}곳"
        if replay.LABEL == replay.LABEL_OBS:      # 실측: 12시부터 선행 강우를 쌓고 화면은 18시부터
            self.start, self.end = replay.REPLAY_FROM, replay.REPLAY_TO
        else:
            self.start, self.end = self.series[0][0], self.series[-1][0]
        self.recompute()

    def recompute(self):
        """처음부터 순서대로 계산 (히스테리시스 상태가 이어지도록). 개구부가 바뀌면 다시 호출."""
        self.engine.reset_state()
        rq = self.p["risk_model"]["confidence"]["rain_quality_replay"]
        times, snaps = [], []
        t = self.start
        while t <= self.end:   # 강우 예보는 이후 실제 값("완벽 예보"), 하수관은 그 시각까지의 실측
            g = {k: [x for x in v if x[0] <= t + timedelta(minutes=60)] for k, v in self.gauges.items()} if self.gauges else None
            snaps.append(self.engine.compute(t, self.series, self.series, self.sewer_obs, rq, live=False, gauge_obs=g))
            times.append(t)
            t += self.cycle
        self.times, self.snaps = times, snaps
        log.info("2022-08-08 재생 %d단계 계산 완료", len(snaps))

    def snap_at(self, step: int) -> tuple[int, Snapshot]:
        step = max(0, min(int(step), len(self.snaps) - 1))
        return step, self.snaps[step]


class LiveRuntime(_Base):
    mode = "live"

    def __init__(self, params: dict | None = None, static=None):
        sewer_api.init_table()
        super().__init__(params, static, sewer_api.load_stations())
        self.lock = threading.Lock()
        self.sewer_obs: dict = {}
        self.sewer_error: str | None = None
        self.version = 0
        self.snap: Snapshot | None = None
        self.prev_grades: np.ndarray | None = None
        self.rain_info: dict = {}
        self.fetch_error: str | None = None
        self.forecast_series: list = []
        self.forecast_info: dict = {}
        seoul_api.init_table()
        self.data_label = (f"실시간: 서울시 강우량계 {'·'.join(seoul_api.GAUGES)} 평균 (10분, "
                           f"{seoul_api.key_label()})")
        self.series: list = []
        self._fetch_live()
        self._fetch_forecast()
        self.next_cycle = datetime.now()
        self._run_cycle(self.next_cycle)

    def _fetch_live(self) -> int:
        """강우 API 수집 → DB 누적 → 시계열 갱신. 실패해도 이전 자료로 계속 판정하되 오류를 화면에 표시."""
        new = 0
        try:
            new = seoul_api.fetch_and_store()
            self.fetch_error = None
        except Exception as e:  # 네트워크·API 오류
            log.warning("강우 수집 실패: %s", e)
            self.fetch_error = str(e)[:200]
        self.last_fetch = datetime.now()
        self.series, self.rain_info = seoul_api.load_series()
        try:
            new += sewer_api.fetch_recent(40)
            self.sewer_error = None
        except Exception as e:
            log.warning("하수관 수위 수집 실패: %s", e)
            self.sewer_error = str(e)[:200]
        self.sewer_obs = sewer_api.load_obs(datetime.now() - timedelta(hours=3))
        return new

    def _fetch_forecast(self):
        """기상청 초단기예보. 실패하면 예측만 비우고 현재 판정은 계속한다."""
        try:
            self.forecast_series, self.forecast_info = kma_api.fetch_forecast()
        except Exception as e:
            log.warning("기상청 예보 수집 실패: %s", e)
            self.forecast_series, self.forecast_info = [], {"error": str(e)[:200]}
        self.last_forecast = datetime.now()

    def _run_cycle(self, t: datetime):
        fseries = kma_api.merge(self.series, self.forecast_series) if self.forecast_series else None
        snap = self.engine.compute(t, self.series, fseries, self.sewer_obs, 0, live=True,
                                   gauge_obs=seoul_api.load_gauges())
        detector.detect_and_notify(self.engine, snap, self.prev_grades, self.p, self.address_full)
        with self.lock:
            self.prev_grades = snap.grades.copy()
            self.snap = snap
            self.version += 1
        rf = snap.rain_now or {}
        subscriptions.save_cycle(t.isoformat(timespec="minutes"), self.mode, rf.get("P60"), rf.get("I10"), None,
                                 json.dumps(self.engine.counts(snap.grades), ensure_ascii=False))

    def refresh_openings(self):
        """개구부가 입력되면 평균 대체값까지 다시 계산하고 바로 재판정."""
        self.engine.set_openings(subscriptions.all_openings())
        self._run_cycle(datetime.now())

    async def loop(self):
        while True:
            await asyncio.sleep(5)
            now = datetime.now()
            try:
                if now - self.last_forecast >= FORECAST_EVERY:
                    await asyncio.to_thread(self._fetch_forecast)
                if now - self.last_fetch >= FETCH_EVERY:
                    new = await asyncio.to_thread(self._fetch_live)
                    # 새 10분 자료가 들어왔거나, 자료가 없어도 주기가 지나면(오래된 자료 → 결측 처리) 재계산
                    if new or now >= self.next_cycle + self.cycle:
                        self.next_cycle = now
                        await asyncio.to_thread(self._run_cycle, now)
            except Exception:
                log.exception("실시간 주기 처리 실패 — 다음 주기에 다시 시도")
