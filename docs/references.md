# 침수위험 판정 로직: 수식과 참고문헌

이 문서는 판정 로직의 각 단계에 쓴 수식과 그 근거 문헌을 정리한다.
문헌은 2026-10-09 Google Scholar·Semantic Scholar 검색으로 찾았고, **초록·검색 요약만 확인**했다(본문 미검토 표시 ※).
고전 이론(SWMM, Manning, Horton, Kalman 등)은 널리 알려진 원전을 적었다.

---

## 반영 수준 구분 (2026-10-10)
**2026-10-11 갱신: SWMM 매뉴얼·Jang(2015)·서울시 기준은 원문을 열어 쪽·식 번호까지 확인해 반영했다 → §10.** 그 밖의 ※ 논문은 여전히 초록만 확인. 로직의 수식은 아래 네 가지 출처로 나뉜다.

### ① 원본(표준) 수식 그대로 — 교과서·원전에 정립된 형태
| 수식 | 원전 | 코드 | 원전과 다른 점 |
|---|---|---|---|
| 비선형 저류 q = (W/A)·√S/n·(d−d_s)^(5/3) | SWMM 매뉴얼(Rossman 2015), Manning | `swmm_surface._reservoir` | SI 단위, 반암시적 풀이. SWMM의 PctZero(웅덩이 없는 포장 비율)·증발 없음 |
| Horton f = f_min + (f₀−f_min)e^(−kt) | Horton 1940 | `swmm_surface.step` | 기본은 젖은 시간 t 방식. `exact: true`면 SWMM 원식(적분형+회복, §10.1) |
| 오리피스 Q = C_d·A·√(2gh) | 수리학 표준(SWMM Vol. II) | `swmm_surface.step` | 식은 같음. 수두 h = (U−1)·H_crit는 우리 가정(④) |
| ARX H(t+1) = aH(t) + bH(t−1) + Σc_k·r(t−k) + d | Ljung 1999 | `risk_model.predict_sewer_arx` | 표준형 |
| 칼만 갱신 Ĥ ← Ĥ + K(H_obs − Ĥ) | Kalman 1960 | `predict_sewer_arx` | **정식 필터 아님**: 매 주기 실측에서 재시작(K=1), 오차 분산 P·R 계산 없음 |
| 로지스틱 회귀, Platt 보정 | 표준 통계, Platt 1999 | `fit_seoul`, `improve` | 표준형 |
| Youden J = TPR − FPR | Youden 1950 | `improve.step_B` | 표준형 |
| Brier, 비용-손실 상대가치 V | Brier 1950, Murphy 1977 | `improve.value`, `step_F` | 표준형 |
| 그래디언트 부스팅, 블록 교차검증 | Friedman 2001, Roberts 외 2017 | `improve.step_D` | scikit-learn 구현 |

### ② 설계 문서(사용자 제공 `basement_flood_risk_logic_prototype.md`)의 식
초과수심 E = max(0, h − z), 4요소 점수표·가중합 R, 등급 20/40/70, override, 신뢰도 가중합, (이전) 하수관 직선 외삽 → v0.2에서 ARX로 교체.

### ③ 검색 논문에서 아이디어만 — 원본 수식 없음 ※
| 논문 | 가져온 것 | 실제로 쓴 방법 |
|---|---|---|
| Jo 외 2026 (SWMM + 앙상블 칼만필터) | 하수관 수위 실측 동화가 예측을 개선 | 앙상블 칼만필터가 아니라 ARX + 실측 재시작 |
| Lund 외 2019, 이소현 외 2023, Huang & Lee 2023 | 하수관 수위를 데이터로 예측 | 그들의 대리모형·ESN이 아니라 ARX |
| Qin 외 2025 | 침수 민감도 + 건물 취약성 결합 | 결합식(로지스틱)과 특성은 우리가 정함 |
| Liu & Mostafavi 2024·2025 | 불균형 자료 처리 필요 | 일반적인 표본 가중 |
| Zischg 외 2018 | 보험·지원금 신고를 정답으로 쓰는 근거 | 그들의 검증 지표는 미사용 |
| Jang 2015, Tian 외 2019, Young 외 2021, Liao 외 2019 | 다중 지속시간 강우 임계값 | ROC·Youden 방식은 우리가 정함, Tabu 탐색 미사용 (결과 미채택) |
| Thiboult 외 2017, Van Houtven 2024, Bischiniotis 외 2019, Zhang 외 2024 | 경보 경계를 비용-손실로 정하는 관점 | 식은 ①의 Murphy 표준식 (결과 미채택) |
| 서울 반지하 취약성 2024, 최유라 2025 | 취약성 특성 선택 근거 | 본문 미검토 → 실제 특성 선택엔 미반영 |
| 레이더 논문 3편 (Lee 2020, Yoon 2023, Zanchetta 2020) | — | 미사용 (자료 신청 필요) |

### ④ 문헌 근거 없는 우리 가정·구성 (근거가 가장 약함)
- 빗물받이 유입 감소: 하수관 이용률 0.8→1.0 사이 직선 감소 `clip((1−U)/0.2, 0, 1)`
- 윗동네 넘침 `r_up × 저지대지수 × 넘친 물` (SWMM RouteTo에서 착안, SWMM 식 아님)
- 도로 흘러내림에 `(1 − 저지대지수)`를 곱해 저지대에 물이 갇히게 함
- 역류 수두 h_s = (U − 1)·H_crit, 만관 수위 H_crit = 수위계별 최고수위(관경 자료 없음)
- 건물별 경계 조정 Δ = θ_i − 40을 20/40/70 모두에 더함
- 강우 비율 점수(B): 비율 0.5에서 0점, 1.5에서 100점 (미채택)
- (예전 경험식) 배수능력 초과 강우만 반영, 저지대 항은 비가 올 때만

### 원문 확보 시 우선 교체할 부분
1. Jo 외 2026 → 실제 앙상블 칼만필터 식·오차 설정 (지금은 실측 재시작 단순형)
2. SWMM 매뉴얼 Vol. I → 누적 침투량 Horton, PctZero, 증발을 원본대로
3. Jang 2015 → 다중 임계값의 실제 결합 규칙으로 B 재시험
4. Qin 외 2025 → 원본 결합 방식·특성 목록과 비교
5. Zischg 외 2018 → 신고 자료 검증 지표를 원본대로

---

## 0. 설계 출발점
- `basement_flood_risk_logic_prototype.md` (사용자 제공 설계 문서): 4요소 점수 가중합, 등급 20/40/70, override, 신뢰도, 보정 절차(§27~48).

## 1. 지표면 물 깊이 — SWMM 물수지 (`src/engine/swmm_surface.py`)
| 수식 | 내용 | 근거 |
|---|---|---|
| dV/dt = Q_runoff + Q_up + Q_back − Q_drain − Q_street, h = V/A_pond | 물수지(연속방정식), 담수 면적 | Rossman, L.A. (2015). *Storm Water Management Model Reference Manual, Vol. I – Hydrology (Revised)*. EPA/600/R-15/162A |
| q = (W/A)·√S/n·(d − d_s)^(5/3) | 비선형 저류 지표 유출 | 위 SWMM 매뉴얼; Manning, R. (1891). On the flow of water in open channels and pipes. *Trans. Inst. Civil Eng. Ireland* 20 |
| f = f_min + (f_0 − f_min)·e^(−kt) | Horton 침투 | Horton, R.E. (1940). An approach toward a physical interpretation of infiltration-capacity. *Soil Sci. Soc. Am. Proc.* 5:399–417 |
| Q_back = C_d·A_o·√(2g·h_s) | 오리피스(맨홀 역류) | 오리피스 식(베르누이·토리첼리), SWMM 매뉴얼 Vol. II(수리) |
| 계수 기본값 | 0.4ha 소유역, 폭 60m, 불투수 75% 등 | 신림동 419 시범 SWMM 모델 `data/replay/sillim419_pilot.inp` |
| 설계 강우 범위 50·75·95mm/h | 서울 방재성능목표 95→100~110mm/h | 서울시 발표(경향신문 2022-09-14, 내 손안에 서울) |

## 2. 문턱 비교·점수·등급 (`src/engine/risk_model.py`)
| 수식 | 근거 |
|---|---|
| E = max(0, h − z_sill), R = Σ wₖ·Sₖ, 등급 20/40/70, override | 설계 문서 §10–17 |
| 가중치 추정 logit P = b₀ + Σ bₖ·Sₖ → 양수 계수 정규화 | 설계 문서 §35; 로지스틱 회귀 표준 |

## 3. 보정 표본·정답 (`src/calibration/seoul_data.py`, `fit_seoul.py`)
| 내용 | 근거 |
|---|---|
| 침수흔적도(재난지원금·보험 신고 포함)를 정답으로, 공간 조인 비율 ≥ 0.1 | Zischg, A.P. 외 (2018). Validation of 2D flood models with insurance claims. *J. Hydrology* 557. doi:10.1016/j.jhydrol.2017.12.042 ※ |
| 시간 순 분리(보정 2010~2022 / 시험 2023~2025), 같은 사건을 나누지 않음 | 설계 문서 §38 |
| 비침수 무작위 표본 + 가중치(전체 비침수 수 ÷ 표본 수) | 층화 표본 가중 추정(표준) |

## 4. 개선 A — 등급 경계를 비용-손실 가치로 (`src/calibration/improve.py: value, step_A`)
- 기대비용 E(θ) = C·(H + F) + L·M, 상대가치 V = (E_무경보 − E)/(E_무경보 − E_완벽), 조건 POD ≥ 0.5
- 지표 POD = H/(H+M), FAR = F/(H+F), CSI = H/(H+M+F), 선행시간 = 피해 시각 − 첫 경보 시각
| 근거 |
|---|
| Murphy, A.H. (1977). The value of climatological, categorical and probabilistic forecasts in the cost-loss ratio situation. *Monthly Weather Review* 105:803–816 |
| Thiboult, A., Anctil, F., Ramos, M.H. (2017). How does the quantification of uncertainties affect the quality and value of flood early warning systems? *J. Hydrology* ※ |
| Van Houtven, G. (2024). Economic value of flood forecasts and early warning systems: a review. *Natural Hazards Review* ※ |
| Bischiniotis, K. 외 (2019). Assessing time, cost and quality trade-offs in forecast-based action for floods. *Int. J. Disaster Risk Reduction* ※ |
| Zhang, R. 외 (2024). Determining the threshold of issuing flash flood warnings based on people's response process simulation. *Hydrol. Earth Syst. Sci.* ※ |
| Acosta-Coll, M. 외 (2018). Real-time early warning system design for pluvial flash floods—A review. *Sensors* ※ |

## 5. 개선 B — 지속시간별 강우 임계값 (`improve.py: step_B, rain_ratio_score`)
- I_d = 최근 d분 강우 ÷ d × 60 (mm/h), d ∈ {10, 30, 60, 180}
- 임계 θ_d = argmax(TPR − FPR) (Youden 지수), 강우 점수 = 100·clip((max_d I_d/θ_d − 0.5)/1.0, 0, 1)
| 근거 |
|---|
| Jang, J.-H. (2015). An advanced method to apply multiple rainfall thresholds for urban flood warnings. *Water* ※ |
| Tian, X., Schleiss, M., Bouwens, C. 외 (2019). Critical rainfall thresholds for urban pluvial flooding inferred from citizen observations. *Sci. Total Environ.* ※ |
| Young, A., Bhattacharya, B. 외 (2021). A rainfall threshold-based approach to early warnings in urban data-scarce regions (Alexandria). *J. Flood Risk Management* ※ |
| Liao, H.Y., Pan, T.Y. 외 (2019). Using Tabu search adjusted with urban sewer flood simulation to improve pluvial flood warning via rainfall thresholds. *Water* ※ |
| Youden, W.J. (1950). Index for rating diagnostic tests. *Cancer* 3(1):32–35 |

## 6. 개선 C — 하수관 수위 예측 (`src/calibration/sewer_forecast.py`)
- ARX: H(t+1) = a·H(t) + b·H(t−1) + Σ_{k=0..5} c_k·r(t−k) + d (수위계별 최소제곱, 2020~2022), 다단계 굴림 예측
- 매 주기 관측으로 상태 갱신 = 관측 오차가 작은 칼만 필터 Ĥ ← Ĥ + K(H_obs − Ĥ), K = P/(P+R) ≈ 1
| 근거 |
|---|
| Jo, J., Duc, L., Sawada, Y. (2026). Real-time monitoring of urban flooding by ensemble Kalman filter (SWMM-DA). *J. Hydroinformatics* ※ |
| Lund, N.S.V., Madsen, H., Mazzoleni, M. 외 (2019). Assimilating flow and level data into an urban drainage surrogate model for forecasting flows and overflows. *J. Hydrology* ※ |
| 이소현, 강동호, 김병식 (2023). 시공간적 강우특성이 반영된 ESN 알고리즘을 활용한 하수관로 수위 변화 예측. 한국수자원학회 학술발표회 ※ |
| Huang, P.C., Lee, K.T. (2023). An alternative for predicting real-time water levels of urban drainage systems. *J. Environmental Management* ※ |
| Kalman, R.E. (1960). A new approach to linear filtering and prediction problems. *J. Basic Engineering* 82:35–45 |
| Ljung, L. (1999). *System Identification: Theory for the User* (2nd ed.). Prentice Hall (ARX 모형) |

## 7. 개선 D — 건물 정적 취약성 (`improve.py: step_D`)
- p_static = GBM(상대고도 100m·300m·1km, 경사, 준공연도, 조적조 여부, 면적, 층수, 과거 침수 횟수(자신·100m 블록, 그 해 이전만))
- 100m 블록 단위 공간 교차검증, 불균형은 표본 가중으로 균형
- 결합 logit P = g₀ + g₁·R/100 + g₂·logit p_static
| 근거 |
|---|
| Qin, X. 외 (2025). Enhancing urban resilience through machine learning-supported flood risk assessment: integrating flood susceptibility with building function vulnerability. *npj Urban Sustainability* ※ |
| Liu, C., Mostafavi, A. (2025). FloodGenome: interpretable machine learning for decoding features shaping property flood risk predisposition in cities. *Environ. Res.: Infrastructure and Sustainability* ※ |
| Liu, C.F., Huang, L., Yin, K., Brody, S., Mostafavi, A. (2024). FloodDamageCast: building flood damage nowcasting with machine-learning and data augmentation. *Int. J. Disaster Risk Reduction* ※ |
| 서울 반지하 가구 침수 취약성 평가 (2024). *한국기후변화학회지* 15(1). doi:10.15531/ksccr.2024.15.1.035 ※ |
| 최유라 (2025). 반지하 주택 침수 영향요인의 공간적 분포 특성 분석: 서울시를 대상으로. *토지주택연구* ※ |
| Roberts, D.R. 외 (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913–929 (블록 교차검증) |
| Friedman, J.H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics* 29:1189–1232; Ke, G. 외 (2017). LightGBM. *NeurIPS* (히스토그램 GBM) |

## 8. 개선 F — 확률 보정 (`improve.py: step_F`)
- P = σ(a·z + b) (보정 기간 적합), Brier = (1/N)·Σ(p − y)², 신뢰도 표(예측 확률 구간별 실제 비율)
| 근거 |
|---|
| Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers* |
| Brier, G.W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1):1–3 |
| 설계 문서 §36 (위험점수와 침수확률 분리) |

## 9. 개선 E — 레이더 강우 (자료 신청 필요, 미실행)
| 근거 |
|---|
| Lee, J.H., Yuk, G.M., Moon, H.T., Moon, Y.I. (2020). Integrated flood forecasting and warning system against flash rainfall in the small-scaled urban stream. *Atmosphere* ※ |
| Yoon, S.S., Shin, H., Heo, J.Y., Choi, K.B. (2023). Assessment of deep learning-based nowcasting using weather radar in South Korea. *Remote Sensing* ※ |
| Zanchetta, A.D.L., Coulibaly, P. (2020). Recent advances in real-time pluvial flash flood forecasting. *Water* ※ |

---

## 실험 결과 기록 (2026-10-09, 서울 전역: 보정 2010~2022 사건 70개 / 시험 2023~2025 사건 18개)
기준 모델 = 서울 보정 SWMM(설계강우 50mm/h, 담수 50%, 넘침 3배) + 회귀 가중치(0.166/0.178/0.641/0.016), 경고 경계 40점.
결과 파일: `data/calibration/seoul/improve/` (report.json, D_F_on_base.json, C_sewer_forecast.json)

| 단계 | 시험 기간 결과 | 채택 |
|---|---|---|
| 기준 | PR-AUC 0.0084, ROC 0.946 / 경고 40점: POD 0.82, 2025-08-13 피해 전 경보 50%, 선행 중앙값 10분 | 기준 |
| A 비용-손실 경계 | C/L 0.05~0.3 모두 60점 선택 → POD 0.39, 피해 전 경보 11% | ✗ (40점 유지) |
| B 강우 임계(10·30·60·180분: 58·47·40·27 mm/h) | 단독 AUC 0.96~0.98이나 결합 점수 PR-AUC 0.0084 → 0.002 | ✗ |
| C 하수관 ARX | 60분 수위 RMSE 0.28 → 0.12 m (관 70% 이상: 0.65 → 0.34 m) | ✅ |
| D 건물 취약성 결합 | PR-AUC 0.0084 → 0.0127 (+51%), ROC 0.956; 같은 경보 규모에서 POD 0.87, 피해 전 경보 74%, 선행 20분 | ✅ |
| F 확률 보정 | Brier 0.00022이나 예측 확률이 실제보다 5~10배 높음(보정기간에 대형 사건 편중) | ✗ (확률 표시 보류) |

반영(v0.2-seoul, `config/settings.yaml`): 서울 보정 SWMM 계수·가중치, C(`risk_model.predict_sewer_arx`), D(`static_prior.parquet` → 건물별 경계 Δ).
2022-08-08 재생 419-1: 19:50 사전 경보(60분 뒤 경고), 20:40 현재 경보 → 보도된 21시대 침수보다 약 1시간 앞섬.

---

## 10. v0.3 — 원문 수식 반영 + 서울시 공공데이터 + 공식 침수예보 결합 (2026-10-10~11)
목표: 현재 운영 중인 서울시 침수예보·경보보다 정확한 건물 단위 경보. 아래는 **원문을 직접 열어 확인한** 위치와 식이다.

### 10.1 원문 위치와 식
| 문헌 (원문 확인) | 위치 | 식 (원문 그대로, 기호만 정리) | 코드 |
|---|---|---|---|
| Rossman, L.A. (2015/2016). *SWMM Reference Manual Vol. I – Hydrology*. EPA/600/R-15/162A (`docs/papers/` 저장, 인쇄 쪽 = PDF 쪽 − 2) | §3.3 Subcatchment Partitioning, p.54, Figure 3-3·3-4 | 소유역 = A1 투수 + A2 불투수(웅덩이 ds) + A3 불투수(웅덩이 0, "% Zero-Imperv") | `swmm_surface.step` exact |
| 〃 | §3.3, p.56, 식 (3-8) | α_P = 1.49·W·S^½ / (A1·n_P) — 투수 소영역 | `swmm_surface.params` exact (SI: 1.49 → 1) |
| 〃 | §3.3, p.56, 식 (3-9) | α_I = 1.49·W·S^½ / ((A2 + A3)·n_I) — 두 불투수 소영역 공통 (W₂/A₂ = W/(A₂+A₃)) | 〃 |
| 〃 | §3.4 Computational Scheme for Runoff, p.57 단계 3b–3e, 4 | d_a = i·Δt + d; (i+f)Δt ≥ d_a 이면 d=0, q=0; i_x = i − e − f; d + i_xΔt ≤ d_s 이면 q=0; Q = Σⱼ qⱼ·Aⱼ | `_reservoir`(i_x 음수 허용), `step` |
| 〃 | §4.2.1, p.88, 식 (4-1) | f_p = f_∞ + (f₀ − f_∞)·e^(−k_d·t) | `horton_exact` |
| 〃 | §4.2.2, p.90, 식 (4-3) | F(t_p) = f_∞·t_p + (f₀ − f_∞)/k_d·(1 − e^(−k_d·t_p)) | `horton_exact` 의 `F` |
| 〃 | §4.2.3 계산 절차, p.94 단계 1–7 | i_a = i + d/Δt; i_a = 0이면 t_p ← −(1/k_d)·ln(1 − e^(−k_rΔt)(1 − e^(−k_d t_p))), f=0; F_p, F₁ (t_p ≥ 16/k_d이면 평탄부); f_p = (F₁ − F_p)/Δt; t₁ > 16/k_d 또는 f_p < i_a 이면 t_p += Δt, 아니면 F_p + i_aΔt = F(t_p)를 Newton–Raphson; f = min(f_p, i_a) | `horton_exact` |
| 〃 | §4.2.4, p.100, 식 (4-14) | k_r = −ln(0.02)/T_dry = 3.912/T_dry (98% 회복) | `params` `kr`, settings `dry_days: 7` |
| Jang, J.-H. (2015). An advanced method to apply multiple rainfall thresholds for urban flood warnings. *Water* 7(11):6056–6078, doi:10.3390/w7116056 (오픈액세스) | §2.3, 식 (1) | RTᵢ = RHᵢ⁰ (RTᵢ < RHᵢ⁰, 비침수 사건) / RTᵢ = RHᵢ¹ (RTᵢ > RHᵢ¹, 침수 사건) — 사건 순서대로 임계 갱신 | `official_compare.jang_run` |
| 〃 | §2.4 및 §3.1 그림 8 workflow | 배수 임계(i ≤ 6h)·유출 임계(i ≥ 12h) 초과 조합 → 초록/노랑(유출만)/주황(배수만)/빨강(둘 다) | 〃 (서울: 10분·1h·3h / 12h) |
| 〃 | §3.1, 식 (2)–(4) | Hit rate = h/(h+m), False alarm rate = f/(f+c), Overall correctness = (h+c)/(h+m+f+c) | `official_compare.gu_metrics` |
| Palmitessa, R., Mikkelsen, P.S., Law, A.W.K., Borup, M. (2021). Data assimilation in hydrodynamic models for system-wide soft sensing and sensor validation for urban drainage tunnels. *J. Hydroinformatics* 23(3):438–452, doi:10.2166/hydro.2020.074 | Methodology › Ensemble-based data assimilation, 식 (1)–(6) (예측, 분석, 칼만 이득, 앙상블 공분산, DEnKF 절반 이득 편차 갱신, 분석 앙상블); 앙상블 100개, 1분마다 갱신 | 식 본문이 이미지라 기호를 옮기지 못함 | **미적용**: 관망 수리모형이 있어야 하는 방법. 우리는 수위계 실측을 상태로 직접 쓰는 ARX(§6)라 동화할 모형 상태가 없음 |
| 서울시 (2023-03-10). 「전국 최초 침수 예·경보제 시행」 news.seoul.go.kr/env/archives/522983 | 본문 침수예보 기준 | 자치구 단위, ① 시간당 55mm 초과 ② 15분 20mm 초과 ③ 도로수위계 침수심 15cm 초과 (2023-05~) | `risk_model.official_rule("seoul2023")` |
| 아시아경제 2026-07-18, 헤럴드경제 2026-05-10 보도 | 침수경보 기준 | 시간당 50mm 이상 + 3시간 90mm 이상; 시간당 72mm 이상이면 즉시 상황판단회의·재난문자 (2026 강화) | `official_rule("seoul2026")` |

원문 추출 시 확인한 점: p.94 단계 3a의 "F₁ = F_p + f₀Δt"는 평탄부라 f_∞가 맞다(SWMM 5 코드 `horton.c`도 f_min) → f_∞로 구현. 회복식은 PDF 텍스트 추출에서 음수 부호가 빠져 보이며, Δt→∞이면 t_p→0, Δt=0이면 t_p 그대로가 되도록 −(1/k_d)로 구현(테스트 `test_horton_exact_recovers_when_dry`). 단계 6의 우변 값은 i_a로 해석(공급이 용량보다 작을 때 공급량만큼 침투).

### 10.2 서울시 공공데이터 (서울 열린데이터광장, 공공누리 1유형) — `data/raw/seoul_open/`
| 자료 | 쓰임 | 코드 |
|---|---|---|
| OA-22715 서울시 강우량 데이터 2006~2025 (강우량계 48곳, 10분) + OA-22824/OA-1168 강우량계 위치 | ① 강우 관측망 AWS 61 → 109곳 (IDW) ② 공식 침수예보 기준을 실제 강우계로 재현 ③ 자치구 발령 판정 | `seoul_gauges.py`, `fit_seoul.RAIN_NET`, `official_compare.gauge_rule_times` |
| OA-21172 풍수해 침수예상도 (침수심 6구간) | 건물 정적 특성 `fmap_depth_m`, `fmap_dist_m` (예상도 안 건물 침수율 18% vs 밖 6.4%) | `static_extra.py` |
| OA-21179 빗물펌프장 공간정보 (118곳) | `pump_dist_m` | 〃 |
| OA-22363 자치구별 불투수면적 (2020) | `gu_imperv_pct` | 〃 |
| OA-21693 자연재해위험개선지구 | 주소 목록뿐(좌표 없음) → 미사용 | — |
| OA-1167 하천 수위 | 최신값만 제공(과거 없음) → 보정 불가, 미사용 | — |

### 10.3 v0.3 판정 로직
1. 강우: 기상청 AWS + 서울시 강우량계 10분 → 건물 위치 IDW(p=2) → 4요소 점수 R (가중치 로지스틱 회귀 재추정: 개구부 0.028·하수관 0.098·강우 0.858·지형 0.016)
2. 건물 경계 θᵢ = 100·(z* − g₀ − g₂·logit pᵢ)/g₁ (g₀ −6.999, g₁ 13.129, g₂ 0.773, z* −2.55; pᵢ = 정적 취약성 GBM, 침수예상도·펌프장·불투수율 특성 추가)
3. **자치구 결합(AND)**: 건물이 속한 자치구의 서울시 강우량계 중 하나라도 침수예보 기준(1h 55mm 또는 15분 20mm 초과)을 최근 3시간 안에 넘으면 G=1.
   경고 이상 = G=1 이고 R ≥ θᵢ + 15 (개구부 초과·급상승 override 포함). G=0이면 최대 '주의'. 강우계 자료가 없으면 G 미적용(v0.2처럼).
   - 근거: 공식 기준은 "어디(구)"를, 건물 모델은 "어느 반지하"를 가려낸다. 둘을 곱하면 구 밖 오경보와 구 안 저위험 건물 경보가 함께 줄어든다.
   - +15는 **보정 기간(2010~2022)만으로** "POD가 서울시 침수경보의 보정기간 POD(0.915) 이상인 가장 작은 경보 규모"로 정함(→ +15, 보정 POD 0.916)
4. 예보(10~60분): 강우계 관측 + 기상청 초단기예보로 G와 R을 미래 시점마다 다시 계산 → 사전 경보

### 10.4 시험 결과 (2023~2025 사건 18개, 건물×사건 270,871행, 침수 949동, 비침수는 표본 가중)
| 방식 | 경보 비율 | POD(적중률) | CSI | 피해 전 경보 | 선행 중앙값 | 오경보 동수/적중 1동 |
|---|---|---|---|---|---|---|
| 서울시 침수예보 (강우계 재현, 구 단위) | 16.2% | 0.940 | 0.0011 | 76% | 40분 | 898 |
| 서울시 침수경보 (50mm+3h 90mm / 72mm) | 4.46% | 0.777 | 0.0033 | 71% | 20분 | 299 |
| Jang(2015) 자치구 임계 갱신 (온라인) | 32.8% | 0.973 | 0.0006 | 81% | 40분 | 1756 |
| v0.2 (운영 중이던 모델) | 7.4% | 0.873 | 0.0023 | 74% | 20분 | 442 |
| **v0.3 (채택: 강우계 추가 + 공공데이터 특성 + 침수예보 AND, +15)** | **4.21%** | **0.881** | **0.0040** | 67% | 20분 | **248** |
| v0.3 같은 방식, 경계 −15 (침수예보와 같은 POD 수준) | 12.3% | 0.936 | 0.0015 | 76% | 40분 | 684 |

- 서울시 침수경보 대비 v0.3: 경보 6% 적게 내면서 적중률 +10%p, CSI +21%, 오경보 −17%. 피해 전 경보는 −5%p (선행 중앙값은 같음)
- 서울시 침수예보 대비 (경계 −15): 경보 24% 적게, 적중률·피해 전 경보·선행시간 거의 같음(−0.4%p·0·0), CSI +36%
- 순위 지표(시험 PR-AUC): v0.2 0.0127 → 공공데이터 특성만 0.0151 → 강우계까지 0.0175 (+38%)
- 자치구 단위(Jang 식 (2)–(4), 구×사건 450개): Jang 적중률 0.75·오경보율 0.26·CSI 0.25 / 침수예보(AWS 보간 근사) 0.35·0.07·0.225
- SWMM 원식(exact): 시험 PR-AUC 0.0084→0.0083(점수), 0.0127→0.0114(결합) — **개선 없음 → 기본값 `exact: false`** (원식 구현은 남겨 둠)
- 한계: 경보 비율·POD는 침수흔적도(신고 기반) 정답 기준. 도로수위계 15cm 조건은 과거 자료가 없어 공식 기준 재현에서 빠짐(공식 체계의 적중률을 약간 낮게 볼 수 있음). 2022-08-08 같은 극한 사건은 보정 기간에 있어 시험엔 없다

### 10.5 재현
```
.venv\Scripts\python -m src.calibration.seoul_gauges            # 강우계 → data/raw/rainfall/10min_seoul_gauge
.venv\Scripts\python -m src.calibration.static_extra            # 침수예상도·펌프장·불투수율 → static_extra.parquet
set FLOOD_RAIN_NET=aws+seoul& set FLOOD_EXTRA_FEATS=1& .venv\Scripts\python -m src.calibration.official_compare dense
set FLOOD_EXTRA_FEATS=1& .venv\Scripts\python -m src.calibration.official_compare rules gauge_rules jang extra compare2
```
결과: `data/calibration/seoul/improve/official_compare.json`, `static_prior_v03.parquet`

## 11. 맑은 날 '주의' 표시 수정 (2026-10-12)
- 문제: 건물별 경계 조정 Δ(§7, §10.3)를 20/40/70 모든 등급 경계에 더해, 취약 건물 1,486동의 '주의' 경계가 2점 이하로 내려갔다.
  비가 0mm여도 지형 점수(가중치 0.016 → 최대 약 1.6점)만으로 주의가 떴다 (2026-10-12 11:30 실시간: 주의 1,393동, 강우 0mm).
  Δ는 '경고' 경계(40점)에 대해서만 보정·검증했으므로 주의 경계에 그대로 쓰는 것은 검증되지 않은 가정이었다.
- 수정 1 (`risk_model.apply_watch_min`, settings `watch_min`): 등급(주의 이상)은 그 시점에 하나라도 있을 때만 —
  60분 강우 ≥ 3mm(기상청 강수강도 '보통 비' 하한), 하수관 이용률 ≥ 0.7(하수관 점수가 오르기 시작하는 값), 개구부 초과 > 0.
  경고 이상은 이미 자치구 침수예보 기준(55mm/h 등)이 필요해 §10.4 시험 결과는 바뀌지 않는다. 2022-08-08 재생 20:40 결과도 동일(경고 39·위험 11,484).
- 수정 2 (settings `vulnerable`): 평상시 취약 건물 = 정적 취약성 p_static 상위 10%(신림동 반지하 1,153동) → 등급 색과 별개로 지도에 보라 테두리, 상세에 '평상시 취약' 표시.
