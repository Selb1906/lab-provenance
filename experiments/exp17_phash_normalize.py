"""실험 17: perceptual hash 전처리(없음 / equalize / autocontrast)에 따른 밝기·대비 등 복원율과 query 단위 오탐.

설계·예측은 notes/개발일지.md "실험 17 설계". 결과는 E:/envs/c2pa-work/exp17/summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
import fphash  # noqa: E402
from exp01_fingerprint import LEVELS, load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp17")
ATK = ["brightness", "contrast", "compression", "blurring", "noise"]
PRE = {"none": lambda im: im, "equalize": lambda im: ImageOps.equalize(im.convert("RGB")),
       "autocontrast": lambda im: ImageOps.autocontrast(im.convert("RGB"), cutoff=1)}


def h(im, pre):
    return fphash.to_hex(fphash.lab_phash(PRE[pre](im.convert("RGB"))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    n = len(items)
    base = {p: [h(im, p) for _, _, im in items] for p in PRE}
    rows = []  # (i, kind, atk, lv, {pre: [dist to each]})
    for i, (name, kind, im) in enumerate(items):
        qs = [("orig", 0.0, im)] + [(a, lv, attacks.apply(im, a, lv, seed=i)) for a in ATK for lv in LEVELS]
        for a, lv, q in qs:
            rows.append((i, kind, a, lv, {p: [fphash.distance(h(q, p), b) for b in base[p]] for p in PRE}))
        print(f"{i + 1}/{n}", flush=True)
    res = {"thresholds": {}, "tpr": {}, "fixed2": {}}
    for p in PRE:
        neg = np.array([min(d for j, d in enumerate(r[4][p]) if j != r[0]) for r in rows])
        t = max([t for t in range(65) if np.mean(neg <= t) <= 0.01], default=-1)
        res["thresholds"][p] = {"t": t, "fpr": float(np.mean(neg <= t)), "fpr_at_2": float(np.mean(neg <= 2))}
        for g in ["figure", "photo"]:
            for a in ATK:
                for T, key in [(t, "tpr"), (2, "fixed2")]:
                    vals = []
                    for lv in LEVELS:
                        rs = [r for r in rows if r[1] == g and r[2] == a and r[3] == lv]
                        vals.append(float(np.mean([r[4][p][r[0]] <= T and r[4][p][r[0]] == min(r[4][p]) for r in rs])))
                    res[key][f"{p}/{g}/{a}"] = vals
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["thresholds"], indent=1))


if __name__ == "__main__":
    main()
