"""C. 하수관 수위 30·60분 예측: 직선 외삽(현행) vs ARX 강우-수위 반응 모형 (Jo 외 2026; Lund 외 2019; 이소현 외 2023).

ARX (수위계 j, 10분 간격):
    H(t+1) = a·H(t) + b·H(t−1) + Σ_{k=0..5} c_k·r(t−k) + d          (최소제곱, 2020~2022 사건일)
다단계 예측: 관측 H(t), H(t−1)에서 시작해 미래 강우를 넣어 1칸씩 굴린다. 매 주기 새 관측으로 상태를 다시 맞추는 것은
관측 오차가 작은 칼만 필터(K≈1)와 같다(Kalman 1960). 미래 강우는 실측을 넣은 '완벽 예보' 상한과 '비 없음' 하한을 함께 본다.
비교: 직선 외삽 H0 + rise30·τ (상한 1.5·H_crit, 현행 risk_model.predict_sewer), 지속 H0.
시험: 2023~2025 사건일. 지표 RMSE(m), 만관 근처(이용률 ≥ 0.7) 구간 RMSE.
실행: .venv\\Scripts\\python -m src.calibration.sewer_forecast
"""
import json
import logging

import numpy as np
import pandas as pd
import pyproj

from src.calibration import seoul_data as sd

log = logging.getLogger(__name__)
OUT = sd.OUT / "improve"
LAGS = 6
_TO5186 = pyproj.Transformer.from_crs(4326, 5186, always_xy=True)


def station_series():
    """수위계별 10분 수위(구간 마지막 값)와 그 위치의 10분 강우(IDW) 표."""
    ss = pd.read_csv(sd.OUT / "sewer_stations.csv").dropna(subset=["lng"]).set_index("station")
    st = sd.stations()
    sx, sy = _TO5186.transform(st.lon.to_numpy(), st.lat.to_numpy())
    qx, qy = _TO5186.transform(ss.lng.to_numpy(), ss.lat.to_numpy())
    W = 1 / np.maximum(np.hypot(qx[:, None] - sx[None, :], qy[:, None] - sy[None, :]), 50) ** 2
    ev = sd.load_events()
    days = sorted({d for s in ev[ev.sewer].days for d in s.split()})
    out = []
    for day in days:
        fs, fr = sd.SEWER / f"{day}.csv", sd.RAIN10 / f"{day}.csv"
        if not (fs.exists() and fr.exists()):
            continue
        s = pd.read_csv(fs)
        s = s[s.station.isin(ss.index)]
        s["t"] = pd.to_datetime(s.ts).dt.floor("10min")
        h = s.groupby(["station", "t"]).level_m.last().unstack(0)
        r = pd.read_csv(fr)
        r["t"] = pd.to_datetime(r.ts) - pd.Timedelta(minutes=10)            # 구간 끝 → 구간 시작 시각
        R = r.pivot_table(index="t", columns="stn", values="rn10").reindex(columns=st.stn, fill_value=np.nan)
        idx = h.index.union(R.index)
        h, R = h.reindex(idx), R.reindex(idx)
        ok = (~R.isna()).to_numpy().astype(float)
        rain = (np.nan_to_num(R.to_numpy()) @ W.T) / np.maximum(ok @ W.T, 1e-12)   # [시각, 수위계]
        rain = pd.DataFrame(rain, index=idx, columns=ss.index)
        for sid in h.columns:
            out.append(pd.DataFrame({"station": sid, "day": day, "t": idx, "H": h[sid].to_numpy(),
                                     "r": rain[sid].to_numpy()}))
    return pd.concat(out, ignore_index=True)


def design(g: pd.DataFrame):
    H, r = g.H.to_numpy(), np.nan_to_num(g.r.to_numpy())
    rows, ys = [], []
    for t in range(LAGS, len(g) - 1):
        x = [H[t], H[t - 1]] + [r[t - k] for k in range(LAGS)] + [1.0]
        if np.isnan(x[:2]).any() or np.isnan(H[t + 1]):
            continue
        rows.append(x); ys.append(H[t + 1])
    return np.array(rows), np.array(ys)


def arx_forecast(coef, H0, H1, rain_future, rain_past, steps):
    h_prev, h = H1, H0
    rp = list(rain_past)                                        # 최근 LAGS개 (최신이 마지막)
    out = []
    for k in range(steps):
        lag = rp[-LAGS:][::-1]
        nxt = coef[0] * h + coef[1] * h_prev + float(np.dot(coef[2:2 + LAGS], lag)) + coef[-1]
        out.append(max(0.0, nxt))
        h_prev, h = h, max(0.0, nxt)
        rp.append(rain_future[k] if k < len(rain_future) else 0.0)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = station_series()
    d["year"] = d.day.str[:4].astype(int)
    hcrit = d[d.year < 2023].groupby("station").H.max()
    pooled_X, pooled_y, coefs = [], [], {}
    for sid, g in d[d.year < 2023].groupby("station"):
        Xs, ys = [], []
        for _, gd in g.groupby("day"):
            X, y = design(gd.sort_values("t").reset_index(drop=True))
            if len(X):
                Xs.append(X); ys.append(y)
        if not Xs:
            continue
        X, y = np.vstack(Xs), np.concatenate(ys)
        pooled_X.append(X); pooled_y.append(y)
        if len(y) >= 200:
            coefs[sid] = np.linalg.lstsq(X, y, rcond=None)[0]
    pooled = np.linalg.lstsq(np.vstack(pooled_X), np.concatenate(pooled_y), rcond=None)[0]
    log.info("ARX 수위계별 %d곳 + 공통 계수, 공통 강우 반응 합 %.4f m/mm", len(coefs), float(pooled[2:2 + LAGS].sum()))
    res = []
    for (sid, day), g in d[d.year >= 2023].groupby(["station", "day"]):
        g = g.sort_values("t").reset_index(drop=True)
        H, r = g.H.to_numpy(), np.nan_to_num(g.r.to_numpy())
        c = coefs.get(sid, pooled)
        hc = hcrit.get(sid, np.nan)
        for t in range(LAGS, len(g) - 6):
            if np.isnan(H[t]) or np.isnan(H[t - 1]) or np.isnan(H[t - 3]):
                continue
            rise30 = (H[t] - H[t - 3]) / 30
            for steps in (3, 6):
                obs = H[t + steps]
                if np.isnan(obs):
                    continue
                lin = H[t] + rise30 * steps * 10
                if not np.isnan(hc):
                    lin = min(max(lin, 0), 1.5 * hc)
                arx_p = arx_forecast(c, H[t], H[t - 1], r[t + 1:t + 1 + steps], r[t - LAGS + 1:t + 1], steps)[-1]
                arx_0 = arx_forecast(c, H[t], H[t - 1], [0.0] * steps, r[t - LAGS + 1:t + 1], steps)[-1]
                res.append((sid, steps * 10, obs, H[t], lin, arx_p, arx_0, H[t] / hc if hc else np.nan))
    R = pd.DataFrame(res, columns=["station", "lead", "obs", "persist", "linear", "arx_perfect", "arx_norain", "util"])
    rm = lambda a, b: float(np.sqrt(np.nanmean((a - b) ** 2)))
    out = {}
    for lead, g in R.groupby("lead"):
        hi = g[g.util >= 0.7]
        out[f"{lead}min"] = {
            "n": len(g), "rmse": {k: round(rm(g[k], g.obs), 4) for k in ("persist", "linear", "arx_perfect", "arx_norain")},
            "rmse_high_util": {k: round(rm(hi[k], hi.obs), 4) for k in ("persist", "linear", "arx_perfect", "arx_norain")},
            "n_high_util": len(hi)}
    out["stations_own_coef"] = len(coefs)
    (OUT / "C_sewer_forecast.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    json.dump({k: list(map(float, v)) for k, v in coefs.items()} | {"_pooled": list(map(float, pooled))},
              open(OUT / "C_arx_coef.json", "w"))
    log.info("C %s", out)


if __name__ == "__main__":
    main()
