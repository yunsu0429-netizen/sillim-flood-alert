"""SQLite 저장소: 구독, 발송 이력, 개구부, 주기 요약."""
import re
import secrets
import sqlite3
import threading
from datetime import datetime

from pathlib import Path

from src.config import settings

DB = Path(settings.db_path)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS subscriptions (
  id INTEGER PRIMARY KEY, bid INTEGER NOT NULL, email TEXT NOT NULL, token TEXT UNIQUE NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', created_at TEXT, confirmed_at TEXT,
  last_sent_at TEXT, last_grade INTEGER, UNIQUE(bid, email));
CREATE TABLE IF NOT EXISTS alerts_sent (
  id INTEGER PRIMARY KEY, sub_id INTEGER, bid INTEGER, grade INTEGER, sim_time TEXT, sent_at TEXT, channel TEXT);
CREATE TABLE IF NOT EXISTS openings (
  id INTEGER PRIMARY KEY, bid INTEGER NOT NULL, kind TEXT, sill_above_ground_m REAL, width_m REAL,
  height_m REAL, barrier_m REAL, source TEXT, note TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS cycles (
  sim_time TEXT PRIMARY KEY, mode TEXT, r60 REAL, i10 REAL, hazard REAL, counts TEXT);
"""


def conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, conn() as c:
        c.executescript(SCHEMA)
        # 예전 DB에 새 열 추가: 건물 고유 ID, 알림 상태(JSON), 개구부 자료 품질
        for table, col in (("subscriptions", "gid"), ("openings", "gid"), ("subscriptions", "alert_state"),
                           ("openings", "quality")):
            cols = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")


# bid(이번 건물 목록의 순번) ↔ gid(건물통합정보 고유 ID). 건물 목록을 갱신하면 bid는 바뀌므로 gid로 다시 연결한다.
_gid_by_bid: list[str] = []


def relink(gids: list[str]) -> dict:
    """서버 시작 시 호출. gid가 비어 있는 예전 행은 현재 bid로 채우고, gid가 있는 행은 새 bid로 옮긴다."""
    global _gid_by_bid
    _gid_by_bid = list(gids)
    bid_by_gid = {g: i for i, g in enumerate(gids)}
    stats = {"filled": 0, "moved": 0, "missing": 0}
    with _lock, conn() as c:
        for table in ("subscriptions", "openings"):
            for r in c.execute(f"SELECT id, bid, gid FROM {table}").fetchall():
                if r["gid"] is None:
                    if 0 <= r["bid"] < len(gids):
                        c.execute(f"UPDATE {table} SET gid=? WHERE id=?", (gids[r["bid"]], r["id"]))
                        stats["filled"] += 1
                elif r["gid"] not in bid_by_gid:
                    stats["missing"] += 1   # 건물이 사라짐 — 행은 남겨 두고 표시만 안 됨
                elif bid_by_gid[r["gid"]] != r["bid"]:
                    c.execute(f"UPDATE {table} SET bid=? WHERE id=?", (bid_by_gid[r["gid"]], r["id"]))
                    stats["moved"] += 1
    return stats


def _gid(bid: int) -> str | None:
    return _gid_by_bid[bid] if 0 <= bid < len(_gid_by_bid) else None


def subscribe(bid: int, email: str) -> dict:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise ValueError("이메일 형식이 올바르지 않습니다")
    with _lock, conn() as c:
        row = c.execute("SELECT * FROM subscriptions WHERE bid=? AND email=?", (bid, email)).fetchone()
        if row and row["status"] == "active":
            return {"status": "already_active", "token": row["token"]}
        token = row["token"] if row else secrets.token_urlsafe(24)
        if not row:
            c.execute("INSERT INTO subscriptions(bid,gid,email,token,created_at) VALUES(?,?,?,?,?)",
                      (bid, _gid(bid), email, token, datetime.now().isoformat(timespec="seconds")))
        return {"status": "pending", "token": token}


def confirm(token: str):
    with _lock, conn() as c:
        row = c.execute("SELECT * FROM subscriptions WHERE token=?", (token,)).fetchone()
        if not row:
            return None
        c.execute("UPDATE subscriptions SET status='active', confirmed_at=? WHERE token=?",
                  (datetime.now().isoformat(timespec="seconds"), token))
        return dict(row)


def unsubscribe(token: str) -> bool:
    with _lock, conn() as c:
        return c.execute("DELETE FROM subscriptions WHERE token=?", (token,)).rowcount > 0


def active_for(bids: list[int]) -> list[dict]:
    if not bids:
        return []
    with _lock, conn() as c:
        q = f"SELECT * FROM subscriptions WHERE status='active' AND bid IN ({','.join('?' * len(bids))})"
        return [dict(r) for r in c.execute(q, bids)]


def all_active() -> list[dict]:
    with _lock, conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM subscriptions WHERE status='active'")]


def save_alert_state(sub_id: int, state_json: str):
    with _lock, conn() as c:
        c.execute("UPDATE subscriptions SET alert_state=? WHERE id=?", (state_json, sub_id))


def subscriber_count(bid: int) -> int:
    with _lock, conn() as c:
        return c.execute("SELECT COUNT(*) FROM subscriptions WHERE status='active' AND bid=?", (bid,)).fetchone()[0]


def mark_sent(sub: dict, grade: int, sim_time: str, channel: str, update_sub: bool = True):
    now = datetime.now().isoformat(timespec="seconds")
    with _lock, conn() as c:
        if update_sub:  # 재생 모드 기록은 실시간 발송 간격 계산에 섞지 않음
            c.execute("UPDATE subscriptions SET last_sent_at=?, last_grade=? WHERE id=?", (sim_time, grade, sub["id"]))
        c.execute("INSERT INTO alerts_sent(sub_id,bid,grade,sim_time,sent_at,channel) VALUES(?,?,?,?,?,?)",
                  (sub["id"], sub["bid"], grade, sim_time, now, channel))


def recent_alerts(limit=20, since: str = "", until: str = "9999") -> list[dict]:
    """판정 시각이 [since, until] 범위인 발송 기록 (실시간/재생 기록이 서로 섞이지 않게)."""
    with _lock, conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT bid, grade, sim_time, sent_at, channel FROM alerts_sent WHERE sim_time >= ? AND sim_time <= ?"
            " ORDER BY id DESC LIMIT ?", (since, until, limit))]


def add_opening(bid: int, d: dict) -> int:
    with _lock, conn() as c:
        cur = c.execute(
            "INSERT INTO openings(bid,gid,kind,sill_above_ground_m,width_m,height_m,barrier_m,source,note,created_at,quality)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (bid, _gid(bid), d.get("kind"), d.get("sill_above_ground_m"), d.get("width_m"), d.get("height_m"),
             d.get("barrier_m"), d.get("source"), d.get("note"), datetime.now().isoformat(timespec="seconds"),
             d.get("quality") or "manual"))
        return cur.lastrowid


def all_openings() -> list[dict]:
    with _lock, conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM openings ORDER BY id")]


def openings_for(bid: int) -> list[dict]:
    with _lock, conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM openings WHERE bid=? ORDER BY id", (bid,))]


def save_cycle(sim_time: str, mode: str, r60, i10, hazard, counts_json: str):
    with _lock, conn() as c:
        c.execute("INSERT OR REPLACE INTO cycles VALUES(?,?,?,?,?,?)", (sim_time, mode, r60, i10, hazard, counts_json))
