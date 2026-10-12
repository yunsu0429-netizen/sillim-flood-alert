// 공개 웹(Vercel 정적 호스팅)용: 서버 API 대신 미리 내보낸 JSON을 읽는다 (src/site/export.py).
//   재생(2022-08-08)  → data/replay/*.json (같은 사이트)
//   실시간            → cfg.live_base (GitHub Actions가 10분마다 갱신하는 live-data 브랜치)
// 메일 알림 신청·개구부 저장처럼 서버가 필요한 기능은 막는다.
(function () {
  const cache = new Map();
  const get = (url, fresh) => {
    if (!fresh && cache.has(url)) return cache.get(url);
    const p = fetch(url, {cache: fresh ? 'no-store' : 'default'}).then(r => {
      if (!r.ok) throw new Error(`자료를 불러오지 못했습니다 (${r.status})`);
      return r.json();
    });
    if (!fresh) { cache.set(url, p); p.catch(() => cache.delete(url)); }
    return p;
  };
  const cfgP = get('data/config.json');
  const minute = () => Math.floor(Date.now() / 60000);
  const live = async name => get(`${(await cfgP).live_base}${name}?t=${minute()}`, true);
  const liveCached = async name => get(`${(await cfgP).live_base}${name}?t=${minute()}`);
  const HORIZONS = [0, 10, 20, 30, 40, 50, 60];
  const GATE_RULE = '침수예보: 1시간 55mm 또는 15분 20mm 초과';

  function reasons(st, s, x, B) {
    const out = [];
    const c = st.current;
    out.push(!c ? '최근 60분 강우 자료가 부족함 → 강우 점수 제외 (0으로 간주하지 않음)'
      : `최근 60분 강우 ${Math.round(c.r60)}mm, 30분 강도 ${Math.round(c.i30)}mm/h`);
    if (s.st < 0) out.push('반경 1000m 안 하수관 수위계 없음 → 하수관 점수 제외');
    else {
      const w = st.sewer[s.st];
      if (!w || w.level_m == null) out.push(`수위계 ${w ? w.station : ''}(${s.sd}m) 최근 자료 없음 → 하수관 점수 제외`);
      else {
        const rise = w.rise30_cm_min != null ? `${w.rise30_cm_min >= 0 ? '+' : ''}${w.rise30_cm_min.toFixed(1)}cm/분` : '상승속도 모름';
        out.push(x.u != null
          ? `하수관 ${w.station}(${s.sd}m) 수위 ${w.level_m.toFixed(2)}m (${rise}), 관 용량 대비 ${Math.round(x.u * 100)}% (기준 ${w.h_crit.toFixed(2)}m: ${w.h_crit_src})`
          : `하수관 ${w.station} 수위 ${w.level_m.toFixed(2)}m`);
      }
    }
    if (x.d[0] != null) out.push(`추정 지표면 물깊이 ${Math.round(x.d[0] * 100)}cm, 개구부 초과 ${Math.round((x.x[0] || 0) * 100)}cm`);
    if (s.rel != null) out.push(`주변 300m 평균보다 ${Math.abs(s.rel).toFixed(1)}m ${s.rel < 0 ? '낮음' : '높음'}`);
    const g = st.gate;
    if (g == null) out.push('자치구 강우량계 자료 없음 → 자치구 기준 없이 건물 기준만으로 판정');
    else out.push(`자치구 강우 기준(${GATE_RULE}) ${g[0] ? '충족 → 건물 기준으로 경고 이상 가능' : '미충족 → 최대 주의'}` +
      (!g[0] && g.some(Boolean) ? ', 60분 안 충족 예상' : ''));
    if (s.dl || g != null) {
      const base = st.level_high + (s.dl || 0);
      out.push(`건물 취약성(과거 침수·지형·구조·침수예상도) 반영 경고 기준 ${Math.round(base)}점` +
        (g != null ? `, 구 발령 중 경고 판정 기준 ${Math.round(base + st.theta_shift)}점` : ''));
    }
    return out;
  }

  async function detail(bid, mode, step) {
    const cfg = await cfgP;
    const all = (await get('data/buildings.json')).buildings;
    const B = all[bid];
    if (!B) throw new Error('건물 없음');
    const c = Math.floor(bid / cfg.chunk);
    const [pack, st] = mode === 'replay'
      ? [await get(`data/replay/detail/${c}.json`), await get(`data/replay/state_${step}.json`)]
      : [await liveCached(`detail/${c}.json`), await live('state.json')];
    const rec = pack[bid];
    const base = {bid, gid: rec?.static.gid ?? null, address: `서울특별시 관악구 신림동 ${B.j}`, road: null,
      use: rec?.static.use ?? B.u, structure: rec?.static.structure ?? '', approved: rec?.static.approved ?? null,
      floors_up: B.fu, floors_down: B.fd, is_banjiha: !!B.b, gl_m: rec?.static.gl ?? null, rel_elev_m: rec?.static.rel ?? null,
      openings: rec?.static.ops ?? [], subscribers: 0, source: rec?.static.src ?? null,
      gl_note: '지반고는 90m 해상도 DEM 보간값으로 실제와 수 m 차이가 날 수 있음'};
    if (!B.b || !rec) return {...base, grade: -1, reasons: ['반지하 후보가 아님 (지하층 없음 또는 비주거)']};
    const s = rec.static, x = rec.steps[mode === 'replay' ? step : 0];
    const t0 = new Date(st.cycle_time);
    const eta = x.tf > 0 ? new Date(t0.getTime() + x.tf * 60000) : null;
    const seven = (a, b) => [a, null, null, null, null, null, b];
    const w = s.st >= 0 ? st.sewer[s.st] : null;
    return {...base, model_version: st.model_version,
      index_label: '침수위험지수 v0.3 (서울 침수흔적도 보정 + 자치구 침수예보 결합, 참고 지표)',
      score: x.s, grade: x.g, raw_level: x.l[0], horizons: HORIZONS, levels: x.l,
      estimated_flood_time_min: x.tf >= 0 ? x.tf : null,
      estimated_flood_clock: eta ? `${String(eta.getHours()).padStart(2, '0')}:${String(eta.getMinutes()).padStart(2, '0')}` : null,
      estimated_protected_flood_time_min: null,
      prediction: {ground_flood_depth_m: seven(x.d[0], x.d[1]), opening_excess_depth_m: seven(x.x[0], x.x[1])},
      components: {opening: [x.c[0]], sewer: [x.c[1]], rain: [x.c[2]], terrain: [x.c[3]]},
      opening: {sill_m: s.sill, barrier_m: s.barrier, quality: s.sq, imputed_from: st.openings.buildings_with_input},
      sewer: w ? {station: w.station, location: w.query, distance_m: s.sd, level_m: w.level_m,
        rise30_cm_min: w.rise30_cm_min, h_crit_m: w.h_crit, h_crit_src: w.h_crit_src, util_now: x.u} : null,
      confidence: {score: x.cf, level: x.cf >= 90 ? 'HIGH' : x.cf >= 70 ? 'MEDIUM' : 'LOW'},
      reasons: reasons(st, s, x, B),
      alert_timeline: mode === 'replay' ? (rec.timeline || []).filter(a => a.t.slice(0, 16) <= st.cycle_time.slice(0, 16)) : null};
  }

  // 점이 건물 안에 있으면 그 건물, 아니면 30m 안 가장 가까운 건물 (서버 _bid_at과 같은 규칙)
  function bidAt(all, lng, lat) {
    const inside = (p, x, y) => {
      let c = false;
      for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
        if ((p[i][1] > y) !== (p[j][1] > y) && x < (p[j][0] - p[i][0]) * (y - p[i][1]) / (p[j][1] - p[i][1]) + p[i][0]) c = !c;
      }
      return c;
    };
    let best = null, bd = Infinity;
    const kx = 111320 * Math.cos(lat * Math.PI / 180), ky = 110540;
    for (const b of all) {
      if (inside(b.p, lng, lat)) return b.id;
      for (const [x, y] of b.p) {
        const d = Math.hypot((x - lng) * kx, (y - lat) * ky);
        if (d < bd) { bd = d; best = b.id; }
      }
    }
    return bd < 30 ? best : null;
  }

  // 네이버 지도 JS의 geocoder 서브모듈 (Client ID만 사용, Secret 불필요) — 도로명·건물명 검색
  async function geocode(q) {
    if (typeof loadNaver !== 'function') throw new Error('지도 모듈 없음');
    await loadNaver();
    const query = q.includes('관악구') ? q : `서울특별시 관악구 ${q}`;
    return new Promise((res, rej) => naver.maps.Service.geocode({query}, (status, r) =>
      status === naver.maps.Service.Status.OK ? res(r.v2.addresses || []) : rej(new Error('네이버 주소 검색 실패'))));
  }

  async function search(q) {
    const all = (await get('data/buildings.json')).buildings;
    const jb = (q.includes('신림동') ? q.split('신림동').pop() : q).replace(/번지/g, '').trim();
    const hit = all.filter(b => b.j === jb);
    const asResult = list => list.map(b => ({bid: b.id, address: `서울특별시 관악구 신림동 ${b.j}`}));
    if (hit.length) return {source: '지번 검색', results: asResult(hit)};
    if ((await cfgP).naver_client_id) {
      try {
        const res = [];
        for (const a of await geocode(q)) {
          const bid = bidAt(all, Number(a.x), Number(a.y));
          if (bid != null && !res.some(r => r.bid === bid)) res.push({bid, address: a.roadAddress || a.jibunAddress});
        }
        if (res.length) return {source: '네이버 주소 검색', results: res};
      } catch (e) { console.warn(e); }
    }
    return {source: '지번 앞부분 검색', results: asResult(all.filter(b => b.j.startsWith(jb)).slice(0, 10))};
  }

  window.STATIC_API = async (path, opts) => {
    if (opts && opts.method && opts.method !== 'GET') throw new Error('공개 웹에서는 지원하지 않는 기능입니다 (알림 신청·개구부 저장은 운영 서버에서만)');
    const u = new URL(path, location.href);
    const p = u.pathname, qs = u.searchParams;
    const mode = qs.get('mode') || 'live', step = Number(qs.get('step') || 0);
    if (p.endsWith('/api/config')) return cfgP;
    if (p.endsWith('/api/buildings')) return get('data/buildings.json');
    if (p.endsWith('/api/state')) {
      if (mode === 'replay') return get(`data/replay/state_${Math.min(step, (await cfgP).replay_times.length - 1)}.json`);
      return live('state.json');
    }
    if (p.endsWith('/api/risk')) return mode === 'replay' ? get(`data/replay/risk_${step}.json`) : live('risk.json');
    const m = p.match(/\/api\/building\/(\d+)$/);
    if (m) return detail(Number(m[1]), mode, step);
    if (p.endsWith('/api/lookup')) {
      const bid = (await get('data/gid.json'))[qs.get('gid')];
      if (bid == null) throw new Error('건물 목록에 없는 건물입니다');
      return {bid};
    }
    if (p.endsWith('/api/search')) return search(qs.get('q') || '');
    throw new Error(`지원하지 않는 요청: ${p}`);
  };
})();
