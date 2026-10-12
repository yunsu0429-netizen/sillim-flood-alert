"""1차 서울 보정에 필요한 자료가 모였는지 확인 (종료 코드 0 = 준비됨).
하수관 기간 침수 사건·큰 침수 사건의 10분 강우 + 하수관 기간 사건일 수위가 모두 있으면 준비 완료."""
import sys

import numpy as np

from src.calibration import seoul_data as sd

e = sd.load_events()
need = e[(e.sewer & (e.trace_polys > 0)) | (e.trace_polys >= 100)]
rain = [d for s in need.days for d in s.split()]
sewer = [d for s in e[e.sewer].days for d in s.split()]
missing = [d for d in rain if not (sd.RAIN10 / f"{d}.csv").exists()] + \
          [d for d in sewer if not (sd.SEWER / f"{d}.csv").exists()]
print(f"남은 날짜 {len(missing)}개")
sys.exit(0 if not missing else 1)
