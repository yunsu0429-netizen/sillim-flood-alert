r"""버전 1(v0.1-prototype)을 v0.2·v0.3과 같은 시험(2023~2025 서울 침수흔적도)으로 평가.
v0.1 = 설계 문서 가정값: 가중치 0.45/0.25/0.20/0.10, SWMM 설계강우 75mm/h·담수 25%·넘침 2배, 경고 40점, 건물별 조정·자치구 결합 없음,
강우는 기상청 AWS 보간. 결과 → data/calibration/seoul/improve/eval_v01.json
실행: .venv\Scripts\python -m src.calibration.eval_v01
"""
import copy
import json
import logging

from src.calibration import fit_seoul as fs
from src.calibration import improve as im
from src.calibration import official_compare as oc
from src.config import load_params

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rm = copy.deepcopy(load_params()["risk_model"])
    rm["weights"] = {"opening": 0.45, "sewer": 0.25, "rain": 0.20, "terrain": 0.10}
    rm["surface"]["swmm"].update({"i_design_mmh": 75, "pond_frac": 0.25, "up_ratio": 2.0, "exact": False})
    D = fs.Data()
    preps = [fs.Prepared(D, e) for _, e in D.events.iterrows()]
    d = im.build(D, preps, rm)
    d.to_parquet(im.OUT / "table_v01.parquet")
    te = d[~d.train]
    out = {"theta40": oc.metrics(te, im.warn_time(te, 40)), "score": im.score_metrics(d, "max_score")}
    (im.OUT / "eval_v01.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("v0.1 %s", out)


if __name__ == "__main__":
    main()
