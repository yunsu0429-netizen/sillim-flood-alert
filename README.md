# 신림동 반지하 침수 위험 3D 대시보드

관악구 신림동(법정동) 건물 21,912동 중 반지하 후보 11,523동(주거용 & 지하층 1개 이상)을
3D 지형 위에 세우고, 강우로 10분마다 계산한 위험도를 색으로 표시한다. 건물별 이메일 알림을 지원한다.

## 실행
```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install geopandas pyogrio shapely pyproj rasterio numpy pandas pyarrow pillow fastapi "uvicorn[standard]" pydantic-settings python-dotenv httpx pyyaml pytest
.venv\Scripts\python -m src.static_data.build_static   # 건물·지형 전처리 (약 1분)
.venv\Scripts\python -m uvicorn src.api.main:app --port 5173
```
http://localhost:5173 (네이버 키에 등록된 URL이 5173 포트라 포트를 바꾸지 말 것)

테스트: `.venv\Scripts\python -m pytest -q`

## 설정 (.env)
- 서버는 실시간과 2022-08-08 재생을 동시에 돌린다. 화면 왼쪽 위 선택칸(실시간 강우 / 2022-08-08 사례)으로 바꿔 보며,
  `MODE`는 처음 열 때의 기본값이다. 재생 모드의 알림은 실제로 보내지 않고 `data/outbox/`에만 남긴다
- `REPLAY_SPEED`: 재생 배속
- `KMA_API_KEY`: 공공데이터포털 「기상청 단기예보 조회서비스」 키. 초단기예보 1시간 강수량(RN1, 신림동 격자 59·125)을
  10분 6칸에 나눠 30·60분 뒤 위험을 계산한다. 범주값(예: "30.0~50.0mm")은 중간값, "1mm 미만"은 0.5mm로 쓴다
- `NAVER_MAP_CLIENT_ID/SECRET`: 주소 검색(Geocoding)과 거리뷰 패널
- `GMAIL_USER`, `GMAIL_APP_PASSWORD`: Google 계정 2단계 인증 → 앱 비밀번호. 비어 있으면 메일을 보내지 않고 `data/outbox/*.eml`로 저장
- `MODE=live`: 실시간 모드. 서울 열린데이터광장 「서울시 강우량 정보」(ListRainfallService)의 관악구 강우계
  신림P(신림동 빗물펌프장)·관악구청 10분 강우량을 2분마다 확인해 평균을 쓴다.
  - `SEOUL_OPEN_API_KEY`가 없으면 공개 sample 키(최근 5건)로 받아 DB에 쌓는다. 이 경우 서버를 켠 뒤 약 10~30분은
    최근 60분 자료가 4개 미만이라 "판정 보류(회색)"로 표시된다. 인증키를 넣으면 시작 즉시 5시간 분량을 받는다
  - 30·60분 예측은 기상청 초단기예보(KMA_API_KEY)로 계산한다

## 모델 v0.3-seoul (현재 설정, 상세 `docs/references.md` §10)
- 강우: 기상청 AWS + 서울시 강우량계(열린데이터광장 OA-22715) 보간, 가중치·건물 경계는 서울 침수흔적도 2010~2022로 보정
- 건물 취약성에 서울시 풍수해 침수예상도·빗물펌프장 거리·자치구 불투수율 추가
- **경고 이상 = 관악구 강우량계가 서울시 침수예보 기준(1시간 55mm 또는 15분 20mm 초과)을 넘었고 건물 점수가 경계+15 이상**
  (`risk_model.district_gate`). 2023~2025 시험에서 서울시 침수경보보다 경보 6% 적게, 적중률 +10%p, CSI +21%
- 아래 v0.1 설명의 4요소 점수·SWMM 지표면 계산은 그대로 쓰며, 계수만 v0.3 값으로 바뀌었다

## 위험 계산 (모델 v0.1-prototype, `basement_flood_risk_logic_prototype.md` §1–23)
`src/engine/risk_model.py`, 계수·임계값은 `config/settings.yaml`의 `risk_model` (모두 보정 전 가정값)
1. 강우: 신림P·관악구청 10분 강우 → P10·P30·P60·P180, I10·I30·I60. 미래는 관측+기상청 초단기예보(재생은 이후 값)
2. 하수관: 서울시 하수관로 수위(관악구 34곳, 건물에서 1km 안 최근접) → H0·H10·H30, 30분 상승속도,
   `H_pred = H0 + 상승속도×분`(상한 1.5×H_crit). 관 용량 H_crit은 자료가 없어 **2022-08-08 최고수위**(7곳)로 가정, 나머지는 그 중앙값
3. 지표면 물깊이 — `surface.method`로 선택
   - **swmm (기본)** `src/engine/swmm_surface.py`: 건물마다 가상 소유역(0.4ha)을 판정 6시간 전부터 +60분까지 5분 간격 적분.
     SWMM 5 식 그대로 — 비선형 저류 유출 `q = (W/A)·√S/n·(d−ds)^(5/3)`, Horton 침투, 우수받이 용량(설계강우 × 하수관 여유
     `clip((1−U)/(1−0.8),0,1)`), 맨홀 역류 오리피스식 `Cd·A·√(2g·h_s)`, 저지대 RouteTo 유입, 담수 `V/A_pond`,
     도로 Manning 흘러내림. 기본 계수는 `data/replay/sillim419_pilot.inp`와 같다. 관망 동역학 대신 수위계 실측을 쓴다
   - empirical: `max(0, α·max(0,H_pred−H_crit) + β·max(0, I30−50mm/h) + γ·저지대·(물 있을 때))` (α 0.5, β 0.01, γ 0.15)
4. 개구부 초과수심 `max(0, 물깊이 − 문턱높이)`, 차수판 반영값 별도. 문턱은 지면 기준 입력값, 없으면 **입력된 건물들의 평균**(평균 대체값)
5. 점수 `0.45·개구부 + 0.25·하수관 + 0.20·강우(60분) + 0.10·지형(상대고도)`, 자료 없는 요소는 빼고 재정규화
6. 등급 20/40/70점 + override(개구부 초과 >0 → 최소 경고, ≥0.30m → 위험, 수위 급상승+호우 예보 → 최소 경고).
   화면 등급은 상승 즉시·하강은 30분 지속 후
7. 신뢰도 `0.30·하수관 + 0.25·강우 + 0.25·지형 + 0.20·개구부 자료` → 높음(90+)/보통(70+)/낮음
8. 지금·+10·…·+60분마다 계산, `T_flood` = 처음 개구부 초과가 생기는 시점

## 알림 시점 (`src/alerts/detector.py` `evaluate`)
- 현재 경보: 화면 등급이 '주의' 이상으로 오를 때
- 사전 경보: 10~60분 예보에서 처음 '경고' 이상이 되거나 개구부 유입이 60분 안에 예상될 때 "N분 뒤(HH:MM) 예상"
- 재발송: 예상 등급 상승, 예상 시각 10분 이상 앞당겨짐, 남은 시간이 처음 20분 이하. 60분간 위험 예보가 없으면 해제
- 같은 주기에 겹치면 한 통으로. 재생 모드는 메일 없이 건물 상세에 "알림 타임라인"으로 표시
  (2022-08-08 419-1: 19:50 첫 사전경보 → 20:00 상향 → 20:30 최종, 실제 침수 21시대)

## 하수관 수위 자료
- `python -m src.sources.sewer_api --stations` : 수위계 위치(주소)를 네이버 지오코딩 → `data/manual/sewer_stations.csv`
- `python -m src.sources.sewer_api --backfill 2022-08-08` : 재생용 실측 → `data/replay/sewer_20220808.csv`
- 실시간은 서버가 2분마다 최근 40분을 받아 SQLite에 누적

## 데이터와 한계
- 건물: 국토교통부 GIS건물통합정보 `AL_D010_11_20260909` (EPSG:5186). 필드 매핑은 `src/static_data/buildings.py` 주석 참고.
  높이는 A16(없으면 층수×3m). A12는 높이가 아니라 건축면적이다
- 지형: 국토지리정보원 공개 DEM 37612 (EPSG:5179, **90m 해상도**). 건물 지반고 오차가 수 m 날 수 있다
  (419-1: DEM 보간 44.8m / study1 38.505m / 빔방울 40.4m)
- 재생 강우는 기상청 AWS 10분 실측(410 기상청·417 금천·509 관악)을 신림동 중심으로 거리가중 합성(`src/sources/replay.py`, 12시부터 선행 강우 적분, 화면은 18시~02시). 파일이 없으면 예전 Huff 3분위 가정 하이토그래프(첨두 130mm/hr)로 대체. 재생 하수관 수위도 실측. 재생의 예보는 이후 실제 값을 쓴 "완벽 예보"다
- 신림동 안 강우계가 1~2개라 강우는 동네 전체에 같게 적용된다. 건물 간 색 차이는 지형 취약도에서 나온다
- 반지하 후보는 "주거용 & 지하층≥1"로 근사한 것이며 실제 반지하 세대 여부와 다를 수 있다
- 배경지도 © OpenStreetMap contributors (타일 이용정책상 개발·시연 규모에서만 사용)

## 클라우드 배포 (Render)
1. 이 폴더를 GitHub 비공개 저장소에 올린다 (`.env`는 제외됨, `data/processed`는 포함)
2. Render → New → Blueprint → 저장소 선택 → `render.yaml`대로 생성
   - `starter` 요금제 + 1GB 디스크: 항상 켜져 있어 실시간 수집·메일 알림이 동작하고, 구독자 DB가 재배포 후에도 유지된다
   - free 요금제는 15분 동안 접속이 없으면 잠들어 그동안 수집·알림이 멈추고, 디스크가 없어 재배포 때 구독자가 지워진다
3. Render 화면에서 비밀값 입력: `SEOUL_OPEN_API_KEY`, `KMA_API_KEY`, `NAVER_MAP_CLIENT_ID/SECRET`, `GMAIL_USER/APP_PASSWORD`,
   `PUBLIC_BASE_URL`(발급된 https 주소)
4. 네이버 클라우드 콘솔 → Maps Application → Web 서비스 URL에 발급된 주소 추가 (안 하면 거리뷰·주소검색 실패)
5. 개구부 입력은 관리자만: Render가 만든 `ADMIN_TOKEN` 값으로 `https://<주소>/?admin=<값>` 한 번 접속

공개 서버 보호: 알림 신청은 IP당 시간 5건·이메일당 하루 3건, 네이버 주소 검색은 IP당 분 30건으로 제한한다.
재생(2022년 사례)은 서버 시작 때 미리 계산하고 브라우저마다 따로 재생되며, 재생 중에는 메일을 보내지 않는다.
배경지도는 OpenStreetMap 공용 타일이라 접속자가 많아지면 VWorld 등 별도 타일 서비스로 바꿔야 한다.

## 지형·건물 데이터 갱신
**5m 지형으로 교체**
1. 국토정보플랫폼(map.ngii.go.kr)에서 신림동을 덮는 **수치표고모형 5m** 도엽들, 또는 **수치지형도(1:5,000) 등고선·표고점**을 받아 `data/raw/dem5m/`에 넣는다
2. `.venv\Scripts\python -m src.static_data.build_static --dem-dir data/raw/dem5m` (또는 `.env`에 `DEM_DIR=data/raw/dem5m`)
   - 래스터 여러 장은 이어 붙이고, 등고선은 5m 격자로 선형 보간한다. 높이 필드를 못 찾으면 로그에 필드 목록이 나오니 `src/static_data/dem.py`의 `HEIGHT_FIELDS`에 추가
   - 로그에서 해상도, 빈 칸 수, 419-1 지반고 비교, 반지하 취약도 V의 다양성(현재 100가지)을 확인
3. 서버 재시작. `DEM_DIR`을 비우면 90m 공개 DEM으로 돌아간다

**건물정보 갱신 (브이월드 + SHP 합치기, 기본값)**
- `.venv\Scripts\python -m src.static_data.build_static` (= `--buildings merged`, 약 3분, `VWORLD_API_KEY` 필요)
- 브이월드 국가공간정보 WFS(`getBldgisSpceWFS`, `dt_d010`)의 GIS건물통합정보를 우선 쓰고, 브이월드에 없는 건물(2026-10 기준 386동:
  용도 없음 254, 서울대 등 교육연구시설 114, 반지하 후보 5)은 SHP에서 보충한다. 필드 매핑은 `config/settings.yaml`의 `vworld.fields`
- 브이월드 요청이 4번 재시도 후에도 실패하면 중단한다 (SHP로 조용히 대체하지 않음). SHP만 쓰려면 `--buildings shp`
- 상세 패널 하단에 건물마다 출처(브이월드 수집일 / SHP)를 표시한다
- 구독·개구부와 대시보드 링크(`?gid=`)는 건물통합정보 고유 ID로 저장되어, 건물 목록이 바뀌어도 같은 건물을 가리킨다
- 확인용: `.venv\Scripts\python -m src.static_data.vworld --probe` (레이어 목록과 419-1 부근 샘플 필드 출력)
