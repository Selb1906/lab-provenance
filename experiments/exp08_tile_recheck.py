"""실험 8: 조각 해시 재평가 — 등록 비율 밖 크롭, 질의 단위 오탐, crop_resistant_hash 비교.

설계·예측은 notes/개발일지.md "실험 8 설계". 결과는 E:/envs/c2pa-work/exp08/.
"""
import json
import sys
from pathlib import Path

import imagehash
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402
from exp05_tile_fingerprint import h64, mind, tiles  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp08")
AREAS_OFF = [0.95, 0.85, 0.75, 0.65, 0.55]   # 등록 비율 사이 값
AREAS_ON = [0.9, 0.8, 0.7, 0.6, 0.5]         # 실험 5와 같은 값(비교용)


def rel_for_area(a):
    # attacks.RANGES["resizedcrop"] = (1, 0.5): 면적 = 1 - 0.5*rel
    return (1 - a) / 0.5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    n = len(items)
    T = [tiles(im) for _, _, im in items]
    C = [imagehash.crop_resistant_hash(im) for _, _, im in items]
    print("registered", n, flush=True)
    rows = []  # (src, kind, area, on/off, tile_d_to_each[n], crh_matches_to_each[n])
    for i, (name, kind, im) in enumerate(items):
        qs = [("orig", 1.0, im)]
        for a in AREAS_OFF:
            qs.append(("off", a, attacks.apply(im, "resizedcrop", rel_for_area(a), seed=i)))
        for a in AREAS_ON:
            qs.append(("on", a, attacks.apply(im, "resizedcrop", rel_for_area(a), seed=i)))
        for tag, a, q in qs:
            hq = h64(q)
            cq = imagehash.crop_resistant_hash(q)
            td = [mind(hq, T[j]) for j in range(n)]
            cm = [cq.hash_diff(C[j])[0] for j in range(n)]
            rows.append((i, kind, tag, a, td, cm))
        print(f"{i + 1}/{n} {name}", flush=True)

    # 음성: 질의를 자기 자신을 뺀 원장과 비교한 최선 점수 / 양성: 자기 자신과의 점수(+가장 가까운 것이 자기인지)
    def neg_tile(r):
        return min(d for j, d in enumerate(r[4]) if j != r[0])

    def neg_crh(r):
        return max(m for j, m in enumerate(r[5]) if j != r[0])

    neg_t = np.array([neg_tile(r) for r in rows])
    neg_c = np.array([neg_crh(r) for r in rows])
    res = {"n_queries_negative": len(rows), "thresholds": {}, "tpr": {}}
    for fpr in [0.01, 0.001]:
        t_tile = max([t for t in range(65) if np.mean(neg_t <= t) <= fpr], default=-1)
        k_crh = min([k for k in range(1, 200) if np.mean(neg_c >= k) <= fpr], default=999)
        res["thresholds"][str(fpr)] = {"tile_t": t_tile, "tile_fpr": float(np.mean(neg_t <= t_tile)),
                                       "crh_k": k_crh, "crh_fpr": float(np.mean(neg_c >= k_crh))}
        for g in ["figure", "photo"]:
            for tag, areas in [("off", AREAS_OFF), ("on", AREAS_ON)]:
                for a in areas:
                    rs = [r for r in rows if r[1] == g and r[2] == tag and r[3] == a]
                    ok_t = [r[4][r[0]] <= t_tile and r[4][r[0]] == min(r[4]) for r in rs]
                    ok_c = [r[5][r[0]] >= k_crh and r[5][r[0]] == max(r[5]) for r in rs]
                    res["tpr"][f"{fpr}/{g}/{tag}/{a}"] = {"tile": float(np.mean(ok_t)), "crh": float(np.mean(ok_c))}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["thresholds"], indent=1))


if __name__ == "__main__":
    main()
