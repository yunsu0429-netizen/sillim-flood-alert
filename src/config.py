from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MANUAL = DATA / "manual"
STUDY1 = ROOT.parents[1] / "study1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    mode: str = "live"  # 처음 열 때 보여줄 화면: live | replay
    replay_speed: float = 60.0
    public_base_url: str = "http://localhost:5173"

    building_shp: str = str(STUDY1 / "AL_D010_11_20260909" / "AL_D010_11_20260909.shp")
    dem_path: str = str(STUDY1 / "(B080)공개DEM_37612_img_2025" / "37612" / "37612.img")
    dem_dir: str = ""                        # 5m DEM 도엽 또는 수치지형도 등고선 폴더. 비우면 dem_path(90m) 사용
    building_source: str = "merged"          # merged(브이월드 우선 + SHP 보충) | vworld | shp
    vworld_api_key: str = ""                 # 전처리(build_static --buildings vworld)에만 필요
    replay_inp: str = str(DATA / "replay" / "sillim419_pilot.inp")  # study1에서 복사 (배포에 포함)
    db_path: str = str(DATA / "app.db")      # 클라우드에서는 영구 디스크 경로로 지정
    admin_token: str = ""                   # 설정하면 개구부 입력에 X-Admin-Token 필요

    naver_map_client_id: str = ""
    naver_map_client_secret: str = ""
    gmail_user: str = ""
    gmail_app_password: str = ""
    seoul_open_api_key: str = ""
    kma_api_key: str = ""
    kma_apihub_key: str = ""             # 기상청 API허브 (칼리브레이션용 과거 AWS 강수)


settings = Settings()


def load_params() -> dict:
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
