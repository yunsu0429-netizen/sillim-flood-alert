"""네이버 클라우드 Maps Geocoding / Reverse Geocoding (서버 측 호출, Secret은 .env에만)."""
import logging
import time

import httpx

from src.config import settings

log = logging.getLogger(__name__)
HOSTS = ["https://maps.apigw.ntruss.com", "https://naveropenapi.apigw.ntruss.com"]  # 신규 / 구 도메인


def enabled() -> bool:
    return bool(settings.naver_map_client_id and settings.naver_map_client_secret)


def _get(path: str, params: dict) -> dict:
    headers = {"x-ncp-apigw-api-key-id": settings.naver_map_client_id,
               "x-ncp-apigw-api-key": settings.naver_map_client_secret, "Accept": "application/json"}
    last = None
    for host in HOSTS:
        r = httpx.get(host + path, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
        last = f"{host} → HTTP {r.status_code} {r.text[:200]}"
        log.warning("네이버 API 실패: %s", last)
    raise RuntimeError(last)


def geocode(query: str) -> list[dict]:
    data = _get("/map-geocode/v2/geocode", {"query": query})
    return [{"address": a.get("jibunAddress") or a.get("roadAddress"), "road": a.get("roadAddress"),
             "lng": float(a["x"]), "lat": float(a["y"])} for a in data.get("addresses", [])]


def _number(land: dict) -> str:
    n1, n2 = land.get("number1") or "", land.get("number2") or ""
    return f"{n1}-{n2}" if n2 else n1


def parse_reverse(data: dict) -> dict:
    """Reverse Geocoding 응답 → {"road": 도로명 주소, "building": 건물명, "jibun": 지번 주소}. 없는 항목은 None.

    roadaddr: area1·2 + 도로명(land.name) + 건물번호(number1-number2), 건물명은 land.addition0(type=building)
    addr: area1·2·3 + (산) + 지번(number1-number2), land.type "2"가 산 지번
    """
    out = {"road": None, "building": None, "jibun": None}
    for res in data.get("results", []):
        reg, land = res.get("region", {}), res.get("land", {})
        area = [reg.get(f"area{i}", {}).get("name") for i in range(1, 4)]
        if res.get("name") == "roadaddr" and land.get("name"):
            out["road"] = " ".join(p for p in [*area[:2], land["name"], _number(land)] if p)
            add0 = land.get("addition0") or {}
            if add0.get("type") == "building" and add0.get("value"):
                out["building"] = add0["value"]
        elif res.get("name") == "addr" and land.get("number1"):
            san = "산 " if land.get("type") == "2" else ""
            out["jibun"] = " ".join(p for p in area if p) + f" {san}{_number(land)}"
    return out


_cache: dict[tuple, tuple[float, dict]] = {}
CACHE_SEC = 24 * 3600  # 같은 건물 반복 호출 방지. 약관상 결과를 DB에 영구 저장하지 않고 메모리에만 둔다


def reverse(lng: float, lat: float) -> dict:
    key = (round(lng, 6), round(lat, 6))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SEC:
        return hit[1]
    data = _get("/map-reversegeocode/v2/gc", {"coords": f"{lng},{lat}", "orders": "roadaddr,addr", "output": "json"})
    res = parse_reverse(data)
    _cache[key] = (time.time(), res)
    return res
