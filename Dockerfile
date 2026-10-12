FROM python:3.13-slim

# 서울시·기상청 자료가 한국 시간(시간대 표기 없음)이라 컨테이너 시간대를 맞춘다. 안 맞추면 9시간 어긋나 모두 "판정 보류"가 된다.
ENV TZ=Asia/Seoul PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src
COPY web ./web
# 전처리 결과(건물 3D·지형)와 재생용 강우만 포함. 1.3GB 원본 SHP·DEM은 넣지 않는다.
COPY data/processed ./data/processed
COPY data/replay ./data/replay
COPY data/manual ./data/manual

EXPOSE 5173
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-5173} --proxy-headers --forwarded-allow-ips='*'"]
