"""알림 시점 판단과 메일 작성.

두 종류의 알림 (규칙은 evaluate()에 모아 두어 실시간 발송과 재생 타임라인이 같은 규칙을 쓴다):
1. 현재 경보 — 화면 등급이 '주의' 이상으로 올라갈 때
2. 사전 경보 — 10~60분 예보에서 '경고' 이상이 처음 예상되거나, 개구부 유입(T_flood)이 60분 안에 예상될 때
   다시 보내는 조건: 예상 등급이 더 높아짐 / 예상 시각이 10분 이상 앞당겨짐 / 남은 시간이 처음으로 20분 이하
   60분 동안 예보가 모두 '경고' 미만이고 유입 예상도 없으면 사전경보 상태를 해제한다.
"""
import json
import logging
from datetime import datetime, timedelta

import numpy as np

from src.alerts import mailer, subscriptions
from src.config import settings
from src.engine.risk_model import LEVELS_KO

log = logging.getLogger(__name__)
ACTIONS = {
    1: "배수구·빗물받이 주변을 확인하고, 차수판을 바로 설치할 수 있게 준비하세요.",
    2: "차수판을 설치하고 귀중품·전기제품을 높은 곳으로 옮기세요. 대피 경로를 확인하세요.",
    3: "물이 들어오기 전에 지상층이나 주변 높은 곳으로 대피하세요. 문이 수압으로 열리지 않을 수 있습니다.",
}


def _iso(t):
    return t.isoformat(timespec="minutes") if t else None


def evaluate(state: dict, t: datetime, display: int, prev_display: int | None, levels: list, horizons: list,
             t_flood: int, p: dict) -> tuple[list[dict], dict]:
    """한 건물·한 시각의 알림 판단. (보낼 알림 목록, 새 상태)를 돌려준다. state는 구독별로 저장되는 dict."""
    a = p["risk_model"]["alerts"]
    min_interval = timedelta(minutes=p["alerts"]["min_interval_minutes"])
    st = dict(state or {})
    out = []
    last_sent = datetime.fromisoformat(st["last_sent"]) if st.get("last_sent") else None

    # 1) 현재 경보
    if prev_display is not None and display > prev_display and display >= p["alerts"]["min_grade"]:
        recent = last_sent and t - last_sent < min_interval and display <= st.get("now_level", -1)
        if not recent:
            out.append({"kind": "now", "level": display})
            st["now_level"] = display

    # 2) 사전 경보 후보: 처음으로 '경고' 이상이 되는 예보 시점, 개구부 유입 예상 시점
    eta, pre_level = None, -1
    for h, lv in zip(horizons, levels):
        if h > 0 and lv >= a["pre_alert_level"]:
            eta, pre_level = t + timedelta(minutes=h), max(lv for hh, lv in zip(horizons, levels) if hh > 0)
            break
    if t_flood is not None and t_flood > 0:
        fe = t + timedelta(minutes=int(t_flood))
        eta = fe if eta is None or fe < eta else eta
        pre_level = max(pre_level, a["pre_alert_level"])
    if eta is not None:
        st["candidate_at"] = _iso(t)
        remaining = (eta - t).total_seconds() / 60
        last_eta = datetime.fromisoformat(st["pre_eta"]) if st.get("pre_eta") else None
        reason = None
        if not st.get("pre_active"):
            reason = "first"
        elif pre_level > st.get("pre_level", -1):
            reason = "higher"
        elif last_eta and (last_eta - eta).total_seconds() / 60 >= a["resend_if_earlier_min"]:
            reason = "earlier"
        elif remaining <= a["final_call_min"] and not st.get("final_sent"):
            reason = "final"
        has_inflow = bool(t_flood and t_flood > 0)
        if reason == "first" and display >= pre_level and not has_inflow:
            # 이미 그 단계라 현재 경보로 충분 — 사전경보는 보내지 않고 상태만 기록
            st.update(pre_active=True, pre_level=pre_level, pre_eta=_iso(eta))
        elif reason:
            out.append({"kind": "pre", "level": pre_level, "eta": _iso(eta), "minutes": round(remaining),
                        "reason": reason, "t_flood": t_flood if has_inflow else None})
            st.update(pre_active=True, pre_level=pre_level, pre_eta=_iso(eta),
                      final_sent=st.get("final_sent") or remaining <= a["final_call_min"])
    elif st.get("pre_active") and st.get("candidate_at"):
        if t - datetime.fromisoformat(st["candidate_at"]) >= timedelta(minutes=a["clear_after_min"]):
            st.update(pre_active=False, pre_level=-1, pre_eta=None, final_sent=False)
    if out:
        st["last_sent"] = _iso(t)
    return out, st


# ---------------------------------------------------------------- 메일
def confirmation_mail(email: str, token: str, address: str):
    base = settings.public_base_url
    text = (f"{address} 침수 위험 알림 신청을 받았습니다.\n\n"
            f"아래 링크를 눌러야 알림이 시작됩니다 (본인 확인):\n{base}/api/confirm?token={token}\n\n"
            f"신청하지 않았다면 이 메일을 무시하세요.\n수신 거부: {base}/api/unsubscribe?token={token}")
    return mailer.send(email, f"[신림동 침수 알림] 알림 신청 확인 - {address}", text)


def alert_mail(sub: dict, address: str, alert: dict, detail: dict, reasons: list[str], t: datetime, conf: tuple):
    base = settings.public_base_url
    link = f"gid={sub['gid']}" if sub.get("gid") else f"bid={sub['bid']}"
    lv = alert["level"]
    table = " | ".join(f"+{h}분 {LEVELS_KO[l] if l >= 0 else '-'}" for h, l in zip(detail["horizons"], detail["level"]))
    if alert["kind"] == "pre":
        head = (f"약 {alert['minutes']}분 뒤({alert['eta'][11:16]}) '{LEVELS_KO[lv]}' 수준의 침수 위험이 예상됩니다.")
        if alert.get("now_level") is not None:
            head = f"현재 위험 단계: '{LEVELS_KO[alert['now_level']]}'\n" + head
        if alert.get("t_flood"):
            head += f"\n반지하 개구부(창문·출입구)로 물이 들어오기 시작할 것으로 예상되는 시각: {alert['eta'][11:16]}"
        if detail["t_protected"] > 0:
            head += f"\n차수판을 설치한 경우 예상 유입 시각: +{detail['t_protected']}분"
        subject = f"[신림동 침수 사전경보] {alert['minutes']}분 뒤 {LEVELS_KO[lv]} - {address}"
    else:
        head = f"지금 반지하 침수 위험이 '{LEVELS_KO[lv]}' 단계로 올라갔습니다 (지수 {detail['score'][0] or 0:.0f}/100)."
        subject = f"[신림동 침수 알림] {LEVELS_KO[lv]} - {address}"
    text = (f"[실시간] {t:%Y-%m-%d %H:%M} 기준\n{address}\n\n{head}\n\n"
            f"예보: {table}\n예측 신뢰도: {conf[1]} ({conf[0]}/100)\n\n근거:\n" + "\n".join(f"- {r}" for r in reasons) +
            f"\n\n행동 요령: {ACTIONS.get(lv, '')}\n\n"
            f"이 값은 과거 침수기록으로 보정하지 않은 '프로토타입 침수위험지수'입니다. 공식 재난문자와 지자체 안내를 우선하세요.\n"
            f"대시보드: {base}/?{link}\n수신 거부: {base}/api/unsubscribe?token={sub['token']}")
    return mailer.send(sub["email"], subject, text)


def detect_and_notify(engine, snap, prev_grades: np.ndarray | None, p: dict, address_of):
    """실시간 주기마다: 구독된 건물만 평가해 메일 발송."""
    from src.engine.risk_model import confidence_level
    subs = subscriptions.all_active()
    sent = 0
    for sub in subs:
        bid = sub["bid"]
        if not 0 <= bid < len(snap.grades) or snap.grades[bid] < 0:
            continue
        state = json.loads(sub.get("alert_state") or "{}")
        prev = None if prev_grades is None else int(prev_grades[bid])
        alerts, new_state = evaluate(state, snap.t, int(snap.grades[bid]), prev, [int(v) for v in snap.levels[:, bid]],
                                     snap.ctx.horizons, int(snap.t_flood[bid]), p)
        if alerts:
            # 같은 주기에 현재·사전 경보가 겹치면 한 통으로 (사전경보 본문에 현재 등급을 함께 적음)
            pre = next((a for a in alerts if a["kind"] == "pre"), None)
            now = next((a for a in alerts if a["kind"] == "now"), None)
            al = pre | {"now_level": now["level"]} if pre and now else (pre or now)
            detail = engine.detail(snap, bid)
            conf = (int(snap.confidence[bid]), str(confidence_level(snap.confidence[bid], p["risk_model"]["confidence"]["levels"])))
            channel = alert_mail(sub, address_of(bid), al, detail, engine.reasons(snap, bid), snap.t, conf)
            subscriptions.mark_sent(sub, al["level"], _iso(snap.t), f"{channel}:{'+'.join(a['kind'] for a in alerts)}")
            sent += 1
        if new_state != state:
            subscriptions.save_alert_state(sub["id"], json.dumps(new_state, ensure_ascii=False))
    if sent:
        log.info("%s 알림 %d건", _iso(snap.t), sent)
    return sent


def timeline(snaps: list, bid: int, p: dict) -> list[dict]:
    """재생용: 같은 규칙으로 이 건물에 언제 어떤 알림이 나갔을지 (실제 발송 없음)."""
    state, prev, out = {}, None, []
    for s in snaps:
        alerts, state = evaluate(state, s.t, int(s.grades[bid]), prev, [int(v) for v in s.levels[:, bid]],
                                 s.ctx.horizons, int(s.t_flood[bid]), p)
        prev = int(s.grades[bid])
        out += [{"t": _iso(s.t)} | a for a in alerts]
    return out
