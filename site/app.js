const {Deck, TerrainLayer, SolidPolygonLayer, PathLayer, FlyToInterpolator} = deck;

const COLORS = [[46, 158, 91], [242, 194, 48], [240, 124, 30], [214, 49, 43]];
const NONE = [154, 154, 154];
const GRADE_KO = ['낮음', '주의', '경고', '위험'];
const hex = c => `rgb(${c.join(',')})`;
const gradeColor = g => (g >= 0 ? COLORS[g] : NONE);
const $ = id => document.getElementById(id);

let cfg, meta, buildings = [], grades = [], scores = [], riskVersion = -1, stateCache, selected = null;
let deckgl, chart, naverLoaded = null;
let sewerStations = [];
let mode = 'live';   // 'live' 실시간 | 'replay' 2022-08-08 사례
// 재생 상태는 브라우저마다 따로 둔다 (다른 사람의 재생 조작과 무관)
const player = {step: 0, playing: false, speed: 60, lastTick: 0};
const q = path => path + (path.includes('?') ? '&' : '?') + `mode=${mode}` + (mode === 'replay' ? `&step=${player.step}` : '');
const adminToken = (() => {
  try {
    const t = new URLSearchParams(location.search).get('admin');
    if (t) localStorage.setItem('adminToken', t);
    return localStorage.getItem('adminToken');
  } catch { return null; }
})();
const banjiha = [], others = [];

async function api(path, opts) {
  if (window.STATIC_API) return window.STATIC_API(path, opts);   // 공개 웹(정적 호스팅): static-adapter.js
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}

// ---------- 3D 장면 ----------
function ring3d(b, dz) {
  return b.p.map(([x, y]) => [x, y, dz]);
}

let terrainLayer = null;

// 지형은 한 번만 만든다. 매번 새 extent 배열로 만들면 타일을 다시 받아 지형이 깜박인다.
function getTerrain() {
  if (terrainLayer) return terrainLayer;
  const [w, s, e, n] = meta.bounds;
  const pad = 0.012;
  terrainLayer = new TerrainLayer({
    id: 'terrain',
    minZoom: 12, maxZoom: window.STATIC_API ? 16 : 17, strategy: 'no-overlap',
    elevationDecoder: {rScaler: 6553.6, gScaler: 25.6, bScaler: 0.1, offset: -10000},
    elevationData: window.STATIC_API ? 'data/terrain/{z}/{x}/{y}.png' : '/api/terrain/{z}/{x}/{y}.png',
    texture: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    extent: [w - pad, s - pad, e + pad, n + pad],
    meshMaxError: 1,
    opacity: 0.8,
    color: [210, 214, 205],
  });
  return terrainLayer;
}

function layers() {
  const terrain = getTerrain();
  const ctx = new SolidPolygonLayer({
    id: 'others', data: others, extruded: true, pickable: true,
    getPolygon: b => ring3d(b, (b.z ?? 0) - 2),
    getElevation: b => b.h + 2,
    getFillColor: [170, 172, 176, 110],
    material: {ambient: 0.5, diffuse: 0.5},
  });
  const above = new SolidPolygonLayer({
    id: 'banjiha-above', data: banjiha, extruded: true, pickable: true,
    getPolygon: b => ring3d(b, (b.z ?? 0) - 2),
    getElevation: b => b.h + 2,
    getFillColor: b => [...gradeColor(grades[b.id]), 200],
    updateTriggers: {getFillColor: `${mode}:${riskVersion}`},
    transitions: {getFillColor: 500},
  });
  const below = new SolidPolygonLayer({
    id: 'banjiha-below', data: banjiha, extruded: true, pickable: true,
    getPolygon: b => ring3d(b, (b.z ?? 0) - b.bd - 2),
    getElevation: b => b.bd + 2.3,
    getFillColor: b => [...gradeColor(grades[b.id]), 255],
    updateTriggers: {getFillColor: `${mode}:${riskVersion}`},
    transitions: {getFillColor: 500},
  });
  const sel = new PathLayer({
    id: 'selected', data: selected ? [selected] : [],
    getPath: b => ring3d(b, (b.z ?? 0) + b.h + 0.5), getColor: [31, 111, 235], getWidth: 3, widthUnits: 'pixels',
  });
  const sewer = new deck.ColumnLayer({
    id: 'sewer', data: sewerStations, pickable: true, diskResolution: 12, radius: 18, extruded: true,
    getPosition: s => [s.lng, s.lat], getElevation: s => 40 + 160 * Math.min(s.util ?? 0, 1.5),
    getFillColor: s => s.util == null ? [120, 120, 120] : s.util >= 1 ? [214, 49, 43] : s.util >= 0.9 ? [240, 124, 30] : s.util >= 0.7 ? [242, 194, 48] : [47, 127, 216],
    updateTriggers: {getFillColor: `${mode}:${riskVersion}`, getElevation: `${mode}:${riskVersion}`},
  });
  return [terrain, ctx, below, above, sewer, sel];
}

function initDeck() {
  const [w, s, e, n] = meta.bounds;
  deckgl = new Deck({
    parent: $('scene'),
    initialViewState: {longitude: (w + e) / 2, latitude: (s + n) / 2, zoom: 14.2, pitch: 55, bearing: -20, maxPitch: 80},
    controller: true,
    layers: layers(),
    getTooltip: ({object}) => object && (object.station ? {
      html: `하수관 수위계 ${object.station}<br>${object.query || ''}<br>수위 ${object.level_m ?? '-'}m · 용량 대비 ${object.util != null ? Math.round(object.util * 100) + '%' : '자료 없음'}<br>${object.last_ts ? object.last_ts.slice(11) + ' 측정' : ''}`,
    } : object.p && {
      html: `신림동 ${object.j}<br>${object.u || ''} · 지상 ${object.fu}층 · 지하 ${object.fd}층` +
        (object.b ? `<br>위험: ${grades[object.id] >= 0 ? GRADE_KO[grades[object.id]] : '데이터 없음'} (${scores[object.id] ?? '-'}점)` : ''),
    }),
    onClick: ({object}) => object && object.p && select(object.id, false),
  });
}

const redraw = () => {
  if (deckgl) deckgl.setProps({layers: layers()});
  recolor2d();
};

function flyTo(b) {
  const xs = b.p.map(p => p[0]), ys = b.p.map(p => p[1]);
  const v = deckgl.getViewports()[0];
  deckgl.setProps({initialViewState: {
    bearing: v.bearing, maxPitch: 80, longitude: (Math.min(...xs) + Math.max(...xs)) / 2, latitude: (Math.min(...ys) + Math.max(...ys)) / 2,
    zoom: 17.5, pitch: 60, transitionDuration: 1200, transitionInterpolator: new FlyToInterpolator(),
  }});
}

// ---------- 상태 폴링 ----------
async function refresh() {
  try {
    const st = await api(q('/api/state'));
    if (st.mode !== mode) return;   // 모드 전환 중 늦게 도착한 응답은 버림
    stateCache = st;
    renderState(st);
    if (st.version !== riskVersion) {
      const r = await api(q('/api/risk'));
      if (st.mode !== mode) return;
      grades = r.grades; scores = r.scores; riskVersion = r.version;
      redraw();
      if (selected) loadDetail(selected.id);
    }
  } catch (e) { console.warn(e); }
}

function loop() {
  // 재생 중이면 배속에 맞춰 10분 단계를 넘긴다 (×60이면 실제 10초 = 재생 10분)
  if (mode === 'replay' && player.playing) {
    const now = Date.now();
    if (now - player.lastTick >= 600000 / player.speed) {
      player.lastTick = now;
      if (player.step >= cfg.replay_times.length - 1) player.playing = false;
      else player.step += 1;
    }
  }
  const wait = mode === 'replay' ? (player.playing ? 250 : 1000) : 15000;
  refresh().finally(() => setTimeout(loop, wait));
}

function renderState(st) {
  const t = new Date(st.sim_time);
  $('simTime').textContent = `${t.getMonth() + 1}/${t.getDate()} ${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}`;
  const c = st.current;
  $('rainNow').textContent = !c ? '강우 자료 부족' : `60분 ${c.r60}mm · 30분강도 ${c.i30}mm/h`;
  if (st.live) {
    const L = st.live, g = Object.entries(L.last_by_gauge || {}).map(([k, v]) => `${k} ${v}mm`).join(', ');
    $('liveStatus').innerHTML = `마지막 관측 ${L.last_ts ? L.last_ts.slice(11) : '-'} (${g || '자료 없음'}) · ` +
      `최근 60분 자료 ${L.steps_last_hour}/6개` +
      (L.steps_last_hour < L.min_steps ? ` <b>— ${L.min_steps}개 이상 쌓이면 판정 시작</b>` : '') +
      `<br>수신 확인 ${L.last_fetch ? L.last_fetch.slice(11) : '-'} · ${L.key}` +
      (L.error ? `<br><span style="color:#b42318">강우 수집 오류: ${L.error}</span>` : '') +
      (L.sewer_error ? `<br><span style="color:#b42318">하수관 수집 오류: ${L.sewer_error}</span>` : '');
  } else {
    $('liveStatus').innerHTML = '';
  }
  $('btnPlay').textContent = player.playing ? '일시정지' : '재생';
  const keys = ['낮음', '주의', '경고', '위험', '데이터 없음'];
  $('counts').innerHTML = keys.map((k, i) =>
    `<div style="background:${hex(i < 4 ? COLORS[i] : NONE)}"><b>${st.counts[k]}</b>${k}</div>`).join('');
  const cf = st.confidence, op = st.openings;
  $('riskSummary').innerHTML = `60분 안 개구부 유입 예상 <b>${st.flood_soon}</b>동 · 신뢰도 높음 ${cf.HIGH} / 보통 ${cf.MEDIUM} / 낮음 ${cf.LOW}` +
    `<br>개구부 입력 ${op.buildings_with_input}동, 나머지는 평균 문턱 ${op.imputed_sill_m != null ? Math.round(op.imputed_sill_m * 100) + 'cm' : '-'}로 대체 · 모델 ${st.model_version}`;
  sewerStations = st.sewer || [];
  const f = st.forecast;
  const src = st.live ? `기상청 초단기예보 ${st.forecast_rain?.base || ''} 발표` : '재생 모드는 이후 실제값 사용';
  $('forecast').textContent = f[30]
    ? `예측(${src}) · +30분 위험 ${f[30].counts['위험']}동, 경고 ${f[30].counts['경고']}동 · +60분 위험 ${f[60].counts['위험']}동, 경고 ${f[60].counts['경고']}동`
    : (st.live ? `30·60분 예측 없음${st.forecast_rain?.error ? ' — ' + st.forecast_rain.error : ''}` : '');
  $('topList').innerHTML = st.top.map(x =>
    `<li data-bid="${x.bid}"><span class="dot" style="background:${hex(gradeColor(x.grade))}"></span>신림동 ${x.jibun} <span class="muted">${x.score}점` +
    `${x.t_flood > 0 ? ` · ${x.t_flood}분 뒤 유입` : x.t_flood === 0 ? ' · 유입 중' : ''}</span></li>`).join('');
  $('alertList').innerHTML = mode === 'replay' ? '<li class="muted">재생 모드에서는 알림을 보내지 않습니다.</li>' : st.alerts.length ? st.alerts.map(a =>
    `<li>${a.sim_time.slice(11)} 신림동 ${buildings[a.bid]?.j ?? a.bid} → ${GRADE_KO[a.grade]} (${a.channel.startsWith('gmail') ? '메일 발송' : '개발용 저장(Gmail 미설정)'}${a.channel.includes('pre') ? ' · 사전경보' : ''})</li>`).join('')
    : '<li class="muted">아직 없음</li>';
  updateChart(st);
}

function updateChart(st) {
  const cur = st.cycle_time.slice(11, 16);
  const fr = st.forecast_rain && st.forecast_rain.times ? st.forecast_rain : {times: [], mm: []};
  const labels = [...st.rain.times, ...fr.times];
  const values = [...st.rain.mm, ...fr.mm];
  const nObs = st.rain.times.length;
  if (!chart) {
    chart = new Chart($('rainChart'), {
      type: 'bar',
      data: {labels, datasets: [{data: values, backgroundColor: []}]},
      options: {
        animation: false, plugins: {legend: {display: false}, tooltip: {callbacks: {label: c => `${c.raw} mm/10분`}}},
        scales: {x: {ticks: {maxTicksLimit: 8, font: {size: 10}}}, y: {title: {display: true, text: 'mm/10분', font: {size: 10}}}},
        onClick: (_, els) => els.length && mode === 'replay' &&
          seekTo(chart.data.labels[els[0].index]),
      },
    });
  }
  chart.data.labels = labels;
  chart.data.datasets[0].data = values;
  const idx = mode === 'replay' ? st.rain.times.indexOf(cur) : nObs - 1;
  // 진한 파랑 = 관측(현재까지), 연한 파랑 = 재생 중 아직 안 온 시각, 주황 = 기상청 예보
  chart.data.datasets[0].backgroundColor = labels.map((_, i) => (i >= nObs ? '#f0a35e' : i <= idx ? '#2f7fd8' : '#bcd3ee'));
  chart.update('none');
}

// ---------- 상세 패널 ----------
async function select(bid, fly = true) {
  selected = buildings[bid];
  redraw();
  if (fly) (view === '2d' ? pan2d(selected) : flyTo(selected));
  await loadDetail(bid);
}

async function loadDetail(bid) {
  const d = await api(q(`/api/building/${bid}`));
  selectedGid = d.gid;
  history.replaceState(null, '', d.gid ? `?mode=${mode}&gid=${d.gid}` : `?mode=${mode}&bid=${bid}`);
  const g = d.grade;
  const pct = v => (v == null ? '-' : `${Math.round(v * 100)}cm`);
  const bar = d.levels ? `<div class="hbar">${d.horizons.map((h, i) =>
    `<div title="+${h}분 ${d.levels[i] >= 0 ? GRADE_KO[d.levels[i]] : '판정 불가'}" style="background:${hex(gradeColor(d.levels[i]))}">${h ? '+' + h : '지금'}</div>`).join('')}</div>` : '';
  const confC = {HIGH: '#17602f', MEDIUM: '#7a5300', LOW: '#b42318'};
  const conf = d.confidence ? `<span class="conf" style="color:${confC[d.confidence.level]};border-color:${confC[d.confidence.level]}">신뢰도 ${({HIGH: '높음', MEDIUM: '보통', LOW: '낮음'})[d.confidence.level]} ${d.confidence.score}</span>` : '';
  const eta = d.estimated_flood_time_min == null ? '60분 안 개구부 유입 예상 없음'
    : d.estimated_flood_time_min === 0 ? '<b style="color:#b42318">지금 개구부 높이를 넘는 물이 예상됨</b>'
    : `<b>약 ${d.estimated_flood_time_min}분 뒤(${d.estimated_flood_clock}) 개구부 유입 예상</b>` +
      (d.estimated_protected_flood_time_min != null && d.opening?.barrier_m ? ` · 차수판 설치 시 ${d.estimated_protected_flood_time_min}분 뒤` : '');
  const pr = d.prediction;
  const opq = {manual: '사람 입력', observed: '현장 실측', ai_estimate: 'AI 육안 추정(신뢰도 낮음)', imputed: `평균 대체값(입력된 ${d.opening?.imputed_from}동 평균)`};
  const sw = d.sewer;
  const tl = d.alert_timeline ? `<div class="box"><b>이 건물 알림 타임라인 (재생, 실제 발송 없음)</b><ul class="reasons">${d.alert_timeline.length ? d.alert_timeline.map(a =>
    `<li>${a.t.slice(11, 16)} ${a.kind === 'pre' ? `사전경보: ${a.minutes}분 뒤(${a.eta.slice(11, 16)}) ${GRADE_KO[a.level]} 예상${a.t_flood ? ' · 개구부 유입' : ''}` : `현재 경보: ${GRADE_KO[a.level]}`}</li>`).join('') : '<li class="muted">이 시각까지 알림 없음</li>'}</ul></div>` : '';
  const ops = d.openings.length ? d.openings.map(o =>
    `<li>${o.kind === 'door' ? '출입구' : '창문'}: 바깥 지면 위 <b>${(o.sill_above_ground_m * 100).toFixed(0)}cm</b>부터 유입 가능` +
    `${o.barrier_m ? ` (차수판 ${o.barrier_m}m)` : ''} <span class="muted small">${o.source || ''}</span></li>`).join('')
    : '<li class="muted">등록된 개구부 없음 — 거리뷰를 보고 입력할 수 있습니다.</li>';
  $('detail').innerHTML = `
    <h3 style="font-size:16px">${d.address}</h3>
    ${d.road && d.road.road ? `<div class="muted small" style="margin:-4px 0 8px">도로명 ${d.road.road}${d.road.building ? ` · ${d.road.building}` : ''}</div>` : ''}
    ${d.is_banjiha ? `<span class="grade" style="background:${hex(gradeColor(g))}">${g >= 0 ? GRADE_KO[g] : '판정 불가'} ${d.score != null ? Math.round(d.score) : ''}</span> ${conf}
      <div class="muted small" style="margin-top:4px">${d.index_label}</div>
      ${bar}<div class="small" style="margin:4px 0">${eta}</div>
      <dl>
        <dt>지표면 물깊이</dt><dd>지금 ${pct(pr.ground_flood_depth_m[0])} → 60분 뒤 ${pct(pr.ground_flood_depth_m[6])}</dd>
        <dt>개구부 초과</dt><dd>지금 ${pct(pr.opening_excess_depth_m[0])} → 60분 뒤 ${pct(pr.opening_excess_depth_m[6])}</dd>
        <dt>개구부 문턱</dt><dd>${pct(d.opening?.sill_m)} <span class="muted small">${opq[d.opening?.quality] || ''}</span></dd>
        <dt>하수관 수위계</dt><dd>${sw ? `${sw.station} (${sw.distance_m}m)<br><span class="small">수위 ${sw.level_m ?? '-'}m · ${sw.rise30_cm_min ?? '-'}cm/분 · 용량 대비 ${sw.util_now != null ? Math.round(sw.util_now * 100) + '%' : '-'}</span><br><span class="muted small">용량 기준 ${sw.h_crit_m}m (${sw.h_crit_src})</span>` : '반경 안 수위계 없음'}</dd>
        <dt>점수 구성</dt><dd class="small">개구부 ${d.components.opening[0] ?? '-'} · 하수관 ${d.components.sewer[0] ?? '-'} · 강우 ${d.components.rain[0] ?? '-'} · 지형 ${d.components.terrain[0] ?? '-'}</dd>
      </dl>` : '<span class="muted">반지하 후보 아님</span>'}
    <ul class="reasons">${d.reasons.map(r => `<li>${r}</li>`).join('')}</ul>
    ${tl}
    <dl>
      <dt>용도</dt><dd>${d.use || '-'} · ${d.structure || ''}</dd>
      <dt>층수</dt><dd>지상 ${d.floors_up}층 · 지하 ${d.floors_down}층</dd>
      <dt>지반고(DEM)</dt><dd>${d.gl_m ?? '-'} m <span class="muted small">(90m DEM 보간)</span></dd>
      <dt>주변 대비</dt><dd>${d.rel_elev_m ?? '-'} m</dd>
      <dt>사용승인</dt><dd>${d.approved || '-'}</dd>
    </dl>
    <div class="box"><b>개구부</b><ul class="reasons">${ops}</ul>
      <button id="btnSv">거리뷰 보고 개구부 입력</button></div>
    ${d.is_banjiha && !cfg.static ? `<div class="box"><b>침수 위험 알림 받기</b>
      <div class="muted small">이 건물의 위험 등급이 올라가면 이메일로 알려드립니다. 확인 메일의 링크를 눌러야 시작됩니다.</div>
      <form id="subForm"><input id="subEmail" type="email" placeholder="이메일 주소" required><button>알림 등록</button></form>
      <div id="subMsg" class="small"></div>
      <div class="muted small">현재 이 건물 알림 구독 ${d.subscribers}명</div></div>` : ''}
    <div class="muted small" style="margin-top:8px">${d.gl_note}</div>
    ${d.source ? `<div class="muted small">건물정보 출처: ${d.source}</div>` : ''}`;
  $('right').classList.remove('hidden');
  $('btnSv').onclick = () => openStreetView(d);
  const f = $('subForm');
  if (f) f.onsubmit = async e => {
    e.preventDefault();
    const msg = $('subMsg');
    try {
      const r = await api('/api/subscribe', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({bid: d.bid, email: $('subEmail').value})});
      msg.style.color = '#17602f'; msg.textContent = r.message;
    } catch (err) { msg.style.color = '#b42318'; msg.textContent = err.message; }
  };
}

// ---------- 2D 네이버 지도 ----------
// 반지하 1.1만 동을 한 번에 네이버 폴리곤으로 만들면 느리므로, 화면 안에 보이는 것만(줌 16 이상) 그린다.
const MIN_ZOOM_2D = 16, MAX_POLYS_2D = 1500;
let selectedGid = null;
let view = '3d', map2d = null, last3d = {pitch: 55, bearing: -20};
const polys2d = new Map();   // bid → naver.maps.Polygon (재사용)
const shown2d = new Set();

function bboxOf(b) {
  if (!b.bbox) {
    const xs = b.p.map(p => p[0]), ys = b.p.map(p => p[1]);
    b.bbox = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }
  return b.bbox;
}

function style2d(b) {
  const sel = selected && selected.id === b.id;
  return {fillColor: hex(gradeColor(grades[b.id])), fillOpacity: 0.75,
          strokeColor: sel ? '#1f6feb' : '#333', strokeWeight: sel ? 4 : 0.6, strokeOpacity: sel ? 1 : 0.5};
}

function render2d() {
  if (view !== '2d' || !map2d) return;
  const zoom = map2d.getZoom();
  const msg = $('map2dMsg');
  if (zoom < MIN_ZOOM_2D) {
    for (const id of shown2d) polys2d.get(id).setMap(null);
    shown2d.clear();
    msg.textContent = '확대하면(줌 16 이상) 반지하 건물이 위험 등급색으로 표시됩니다.';
    msg.classList.remove('hidden');
    return;
  }
  const bd = map2d.getBounds(), sw = bd.getMin(), ne = bd.getMax();
  const want = new Set();
  for (const b of banjiha) {
    const [x0, y0, x1, y1] = bboxOf(b);
    if (x1 < sw.x || x0 > ne.x || y1 < sw.y || y0 > ne.y) continue;
    want.add(b.id);
    if (want.size >= MAX_POLYS_2D) break;
  }
  for (const id of shown2d) if (!want.has(id)) { polys2d.get(id).setMap(null); shown2d.delete(id); }
  for (const id of want) {
    const b = buildings[id];
    let pg = polys2d.get(id);
    if (!pg) {
      pg = new naver.maps.Polygon({paths: [b.p.map(([x, y]) => new naver.maps.LatLng(y, x))], clickable: true, ...style2d(b)});
      naver.maps.Event.addListener(pg, 'click', () => select(id, false));
      polys2d.set(id, pg);
    } else {
      pg.setOptions(style2d(b));
    }
    if (!shown2d.has(id)) { pg.setMap(map2d); shown2d.add(id); }
  }
  msg.textContent = want.size >= MAX_POLYS_2D ? `건물이 많아 ${MAX_POLYS_2D}동까지만 표시합니다. 더 확대하세요.` : '';
  msg.classList.toggle('hidden', want.size < MAX_POLYS_2D);
}

const sewer2d = new Map();   // 수위계 → naver.maps.Circle
const sewerColor = s => s.util == null ? '#787878' : s.util >= 1 ? '#d6312b' : s.util >= 0.9 ? '#f07c1e' : s.util >= 0.7 ? '#f2c230' : '#2f7fd8';

function recolor2d() {
  if (view !== '2d' || !map2d) return;
  for (const id of shown2d) polys2d.get(id).setOptions(style2d(buildings[id]));
  const keep = new Set();
  for (const st of sewerStations) {
    keep.add(st.station);
    let c = sewer2d.get(st.station);
    if (!c) {
      c = new naver.maps.Circle({map: map2d, center: new naver.maps.LatLng(st.lat, st.lng), radius: 12, strokeWeight: 2,
                                 strokeColor: '#fff', fillOpacity: 0.95, clickable: true});
      sewer2d.set(st.station, c);
    }
    c.setOptions({fillColor: sewerColor(st)});
    c.__info = st;
  }
  for (const [id, c] of sewer2d) if (!keep.has(id)) { c.setMap(null); sewer2d.delete(id); }
}

function centerOf(b) {
  const [x0, y0, x1, y1] = bboxOf(b);
  return new naver.maps.LatLng((y0 + y1) / 2, (x0 + x1) / 2);
}

function pan2d(b) {
  if (!map2d) return;
  if (map2d.getZoom() < 18) map2d.setZoom(18, false);
  map2d.panTo(centerOf(b));
}

async function setView(v) {
  if (v === view) return;
  const btn2d = document.querySelector('#viewSwitch [data-view="2d"]');
  if (v === '2d') {
    try { await loadNaver(); } catch (e) {
      btn2d.title = e.message;
      alert(`네이버 지도를 열 수 없습니다: ${e.message}`);
      return;
    }
    const vp = deckgl.getViewports()[0];
    last3d = {pitch: vp.pitch, bearing: vp.bearing};   // 3D 화면을 숨기면 뷰포트를 읽을 수 없으므로 미리 저장
    $('map2dWrap').classList.remove('hidden');
    $('scene').classList.add('hidden');
    const center = new naver.maps.LatLng(vp.latitude, vp.longitude);
    const zoom = Math.max(MIN_ZOOM_2D, Math.round(vp.zoom + 1));
    if (!map2d) {
      // 오른쪽 상세 패널에 가리지 않게 확대 버튼을 왼쪽(왼쪽 패널 바로 옆)에 둔다
      map2d = new naver.maps.Map('map2d', {center, zoom, scaleControl: true,
                                            zoomControl: true, zoomControlOptions: {position: naver.maps.Position.LEFT_CENTER}});
      naver.maps.Event.addListener(map2d, 'idle', render2d);
    } else {
      map2d.setCenter(center);
      map2d.setZoom(zoom, false);
    }
    view = '2d';
    if (selected) pan2d(selected);
    render2d();
    recolor2d();
  } else {
    const c = map2d.getCenter();
    $('map2dWrap').classList.add('hidden');
    $('map2dMsg').classList.add('hidden');
    $('scene').classList.remove('hidden');
    view = '3d';
    deckgl.setProps({initialViewState: {longitude: c.x, latitude: c.y, zoom: map2d.getZoom() - 1,
                                        pitch: last3d.pitch, bearing: last3d.bearing, maxPitch: 80}});
    redraw();
  }
  for (const b of document.querySelectorAll('#viewSwitch button')) b.classList.toggle('on', b.dataset.view === view);
}

// ---------- 네이버 거리뷰 ----------
function loadNaver() {
  if (!cfg.naver_client_id) return Promise.reject(new Error('네이버 지도 키가 설정되지 않았습니다'));
  if (!naverLoaded) naverLoaded = new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = `https://oapi.map.naver.com/openapi/v3/maps.js?ncpKeyId=${cfg.naver_client_id}&submodules=panorama,geocoder`;
    // panorama 서브모듈은 본 스크립트 로드 후 따로 받아지므로 생성자가 생길 때까지 기다린다
    s.onload = () => {
      const t0 = Date.now();
      const wait = () => {
        if (window.naver?.maps?.Panorama && window.naver?.maps?.Service) return res();
        if (Date.now() - t0 > 8000) return rej(new Error('네이버 거리뷰 모듈을 불러오지 못했습니다 (키의 Dynamic Map 사용 설정과 등록 URL을 확인하세요)'));
        setTimeout(wait, 100);
      };
      wait();
    };
    s.onerror = () => rej(new Error('네이버 지도 스크립트 로드 실패'));
    document.head.appendChild(s);
  });
  return naverLoaded;
}

async function openStreetView(d) {
  $('svModal').classList.remove('hidden');
  $('svTitle').textContent = `거리뷰 · ${d.address}`;
  $('openingMsg').textContent = '';
  const b = buildings[d.bid];
  const lng = b.p.reduce((a, p) => a + p[0], 0) / b.p.length, lat = b.p.reduce((a, p) => a + p[1], 0) / b.p.length;
  try {
    await loadNaver();
    $('pano').innerHTML = '';
    const pano = new naver.maps.Panorama('pano', {position: new naver.maps.LatLng(lat, lng), pov: {pan: 0, tilt: -5, fov: 100}});
    // 파노라마는 건물에서 가장 가까운 도로 위 촬영점에 열리므로, 그 지점에서 건물 쪽으로 방향을 돌린다
    naver.maps.Event.once(pano, 'pano_changed', () => {
      const p = pano.getPosition();
      const dy = lat - p.y, dx = (lng - p.x) * Math.cos(p.y * Math.PI / 180);
      const pan = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
      pano.setPov({pan, tilt: -5, fov: 100});
      const dist = Math.round(Math.hypot(dx, dy) * 111320);
      $('svTitle').textContent = `거리뷰 · ${d.address} · 촬영점에서 약 ${dist}m`;
      window.__pano = pano;
    });
  } catch (e) {
    $('pano').innerHTML = `<div style="padding:20px">${e.message}</div>`;
  }
  $('openingForm').classList.toggle('hidden', !!cfg.static);   // 공개 웹은 보기만 (저장은 운영 서버)
  if (cfg.static) $('svHint').textContent = '공개 웹에서는 거리뷰 보기만 가능합니다 (개구부 입력은 운영 서버). 이미지는 저장하거나 자동 분석하지 않습니다.';
  $('openingForm').onsubmit = async e => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const num = k => (fd.get(k) === '' ? null : Number(fd.get(k)));
    try {
      await api(`/api/building/${d.bid}/openings`, {method: 'POST',
        headers: {'Content-Type': 'application/json', ...(adminToken ? {'X-Admin-Token': adminToken} : {})},
        body: JSON.stringify({kind: fd.get('kind'), sill_above_ground_m: num('sill'), width_m: num('width'),
          barrier_m: num('barrier'), note: fd.get('note') || null})});
      $('openingMsg').textContent = '저장했습니다.';
      loadDetail(d.bid);
    } catch (err) { $('openingMsg').textContent = err.message; }
  };
}

function seekTo(hhmm) {
  const i = cfg.replay_times.indexOf(hhmm);
  if (i >= 0) { player.step = i; refresh(); }
}

// ---------- 모드 선택 (실시간 / 2022년 사례) ----------
function applyModeUi() {
  const live = mode === 'live';
  for (const btn of document.querySelectorAll('#modeSwitch button')) btn.classList.toggle('on', btn.dataset.mode === mode);
  const badge = $('modeBadge');
  badge.textContent = cfg.modes[mode].label;
  badge.classList.toggle('live', live);
  document.querySelector('.controls').classList.toggle('hidden', live);
  $('chartHint').textContent = live
    ? '최근 6시간 10분 강우량 (파랑: 신림P·관악구청 관측 평균, 주황: 기상청 예보)'
    : '막대를 클릭하면 그 시각으로 이동합니다.';
}

async function setMode(m) {
  if (m === mode) return;
  mode = m;
  riskVersion = -1;
  stateCache = null;
  player.playing = false;
  if (chart) { chart.destroy(); chart = null; }
  applyModeUi();
  history.replaceState(null, '', selected && selectedGid ? `?mode=${mode}&gid=${selectedGid}` : `?mode=${mode}`);
  await refresh();
}

// ---------- 검색·컨트롤 ----------
function bindUi() {
  for (const b of document.querySelectorAll('#viewSwitch button')) b.onclick = () => setView(b.dataset.view);
  $('btnPlay').onclick = () => {
    if (!player.playing && player.step >= cfg.replay_times.length - 1) player.step = 0;
    player.playing = !player.playing;
    player.lastTick = Date.now();
    refresh();
  };
  $('btnReset').onclick = () => { player.step = 0; player.playing = false; refresh(); };
  $('speed').onchange = e => { player.speed = Number(e.target.value); };
  $('closeRight').onclick = () => { $('right').classList.add('hidden'); selected = null; redraw(); };
  $('svClose').onclick = () => $('svModal').classList.add('hidden');
  $('topList').onclick = e => { const li = e.target.closest('li'); if (li) select(Number(li.dataset.bid)); };
  $('searchForm').onsubmit = async e => {
    e.preventDefault();
    const q = $('searchInput').value.trim();
    if (!q) return;
    const box = $('searchResults');
    try {
      const r = await api(`/api/search?q=${encodeURIComponent(q)}`);
      box.innerHTML = r.results.length
        ? r.results.map(x => `<div data-bid="${x.bid}">${x.address}</div>`).join('') + `<span class="muted small">${r.source}</span>`
        : `<span class="muted small">결과 없음 (${r.source}${r.error ? ': ' + r.error : ''})</span>`;
      if (r.results.length === 1) select(r.results[0].bid);
    } catch (err) { box.textContent = err.message; }
  };
  $('searchResults').onclick = e => { const d = e.target.closest('div[data-bid]'); if (d) select(Number(d.dataset.bid)); };
}

async function main() {
  cfg = await api('/api/config');
  const params = new URLSearchParams(location.search);
  mode = cfg.modes[params.get('mode')] ? params.get('mode') : cfg.default_mode;
  if (mode === 'replay' && params.get('step')) player.step = Math.max(0, Number(params.get('step')) || 0);   // 재생 시점 링크
  for (const btn of document.querySelectorAll('#modeSwitch button')) btn.onclick = () => setMode(btn.dataset.mode);
  applyModeUi();
  $('notice').textContent = cfg.notice;
  if (cfg.streamlit_url) $('notice').insertAdjacentHTML('beforeend',
    ` · <a href="${cfg.streamlit_url}" target="_blank" rel="noopener">Streamlit 버전(판정 서버)</a>`);
  const data = await api('/api/buildings');
  meta = data.meta; buildings = data.buildings;
  for (const b of buildings) (b.b ? banjiha : others).push(b);
  $('banjihaCount').textContent = `${meta.banjiha.toLocaleString()}동 / 전체 ${meta.count.toLocaleString()}동`;
  grades = new Array(buildings.length).fill(-1);
  initDeck();
  bindUi();
  await refresh();
  setTimeout(loop, 1000);
  // 링크는 건물 고유 ID(gid)로 공유한다. 예전 ?bid= 링크도 열리지만 데이터 갱신 뒤에는 다른 건물일 수 있다.
  const gid = params.get('gid'), bid = params.get('bid');
  if (gid) {
    try { select((await api(`/api/lookup?gid=${encodeURIComponent(gid)}`)).bid); }
    catch (e) { alert(e.message); }
  } else if (bid) select(Number(bid));
}

main().catch(e => { document.body.insertAdjacentHTML('beforeend', `<pre style="position:absolute;top:40%;left:40%">${e.message}</pre>`); });
