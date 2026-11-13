"""실험 9: 사이트가 쓰는 lab-phash 전체 해시 문턱을 질의 단위 오탐으로 다시 정한다.

설계·예측은 notes/개발일지.md "실험 9 설계". 결과는 E:/envs/c2pa-work/exp09/.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
import fphash  # noqa: E402
from exp01_fingerprint import LEVELS, load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp09")


def h(im):
    return fphash.to_hex(fphash.lab_phash(im.convert("RGB")))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    n = len(items)
    base = [h(im) for _, _, im in items]
    pair = [fphash.distance(base[i], base[j]) for i in range(n) for j in range(i + 1, n)]
    rows = []  # (src, kind, attack, level, dists_to_all)
    for i, (name, kind, im) in enumerate(items):
        qs = [("orig", 0.0, im)] + [(a, lv, attacks.apply(im, a, lv, seed=i)) for a in attacks.RANGES for lv in LEVELS]
        for a, lv, q in qs:
            hq = h(q)
            rows.append((i, kind, a, lv, [fphash.distance(hq, b) for b in base]))
        print(f"{i + 1}/{n} {name}", flush=True)

    neg = np.array([min(d for j, d in enumerate(r[4]) if j != r[0]) for r in rows])
    pair = np.array(pair)
    res = {"n_images": n, "n_queries": len(rows), "pair_threshold_1pct": int(max(t for t in range(65) if np.mean(pair <= t) <= 0.01)),
           "pair_fpr_at": {t: float(np.mean(pair <= t)) for t in range(0, 21)},
           "query_fpr_at": {t: float(np.mean(neg <= t)) for t in range(0, 21)}, "thresholds": {}, "tpr": {}}
    for fpr in [0.01, 0.001]:
        t = max([t for t in range(65) if np.mean(neg <= t) <= fpr], default=-1)
        res["thresholds"][str(fpr)] = {"t": t, "fpr": float(np.mean(neg <= t))}
    for T in range(0, 17):  # 원장 크기마다 맞는 문턱이 달라서 전 구간을 남긴다(2차 실행, 11-13 22:00)
        for g in ["all", "figure", "photo"]:
            for a in ["orig", *attacks.RANGES]:
                for lv in ([0.0] if a == "orig" else LEVELS):
                    rs = [r for r in rows if r[2] == a and r[3] == lv and (g == "all" or r[1] == g)]
                    ok = [r[4][r[0]] <= T and r[4][r[0]] == min(r[4]) for r in rs]
                    res["tpr"][f"{T}/{g}/{a}/{lv}"] = float(np.mean(ok))
    (OUT / "rows.json").write_text(json.dumps([(r[0], r[1], r[2], r[3], r[4]) for r in rows]), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["pair_threshold_1pct", "thresholds"]}, indent=1))


if __name__ == "__main__":
    main()
