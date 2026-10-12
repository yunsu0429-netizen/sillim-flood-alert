"""Stage A rule-based 위험 수식. 가중치·임계값은 config/settings.yaml (모두 가정값)."""
import math

GRADES = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
GRADES_KO = ["낮음", "주의", "경고", "위험"]


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def vulnerability(p: dict, is_banjiha: bool, rel_elev_m=None, slope_pct=None,
                  stream_dist_m=None, flood_count=None) -> float:
    """정적 취약도 V (0~1). 값이 None인 항목은 빼고 남은 가중치로 재정규화한다."""
    if not is_banjiha:
        return 0.0
    v = p["vulnerability"]
    terms = []
    if rel_elev_m is not None and not math.isnan(rel_elev_m):
        terms.append((v["w_lowland"], clip(-rel_elev_m / v["lowland_full_m"], 0, 1)))
    if flood_count is not None:
        terms.append((v["w_history"], min(flood_count / 2, 1)))
    if stream_dist_m is not None:
        terms.append((v["w_stream"], math.exp(-stream_dist_m / v["stream_decay_m"])))
    if slope_pct is not None and not math.isnan(slope_pct):
        terms.append((v["w_flatness"], 1 - clip(slope_pct / v["slope_full_pct"], 0, 1)))
    wsum = sum(w for w, _ in terms)
    if wsum == 0:
        return float("nan")
    return sum(w * x for w, x in terms) / wsum


def rain_hazard(p: dict, r60_mm, i10_mm_hr):
    """H_rain = clip(w60·R60/95 + w10·I10/120, 0, cap). 입력이 None이면 None (0으로 대체하지 않음)."""
    if r60_mm is None or i10_mm_hr is None:
        return None
    r = p["rain"]
    return clip(r["w_r60"] * r60_mm / r["design_mm_per_hr"] + r["w_i10"] * i10_mm_hr / r["i10_ref_mm_per_hr"],
                0, r["h_cap"])


def score(p: dict, hazard, vuln):
    if hazard is None or vuln is None or math.isnan(vuln):
        return None
    base = p["score"]["base_weight"]
    return 100 * min(hazard, 1) * (base + (1 - base) * vuln)


def grade_of(p: dict, s) -> int:
    if s is None:
        return -1
    g = 0
    for i, t in enumerate(p["score"]["thresholds"]):
        if s >= t:
            g = i + 1
    return g


class Hysteresis:
    """등급 상승은 즉시, 하강은 (현재 등급 하한 - hysteresis_points) 미만이 hysteresis_minutes 지속될 때 한 단계."""

    def __init__(self, p: dict):
        self.p = p
        self.state: dict[int, tuple[int, float | None]] = {}

    def update(self, key: int, s, now_min: float) -> int:
        raw = grade_of(self.p, s)
        cur, below_since = self.state.get(key, (raw, None))
        if raw < 0:
            self.state[key] = (-1, None)
            return -1
        if cur < 0 or raw >= cur:
            self.state[key] = (raw, None)
            return raw
        floor = self.p["score"]["thresholds"][cur - 1] - self.p["score"]["hysteresis_points"]
        if s >= floor:
            self.state[key] = (cur, None)
            return cur
        below_since = now_min if below_since is None else below_since
        if now_min - below_since >= self.p["score"]["hysteresis_minutes"]:
            cur, below_since = cur - 1, now_min
        self.state[key] = (cur, below_since)
        return cur
