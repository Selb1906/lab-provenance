"""실험 19: 늘리지 않는(가로세로비 유지, 크기 그대로) crop에서 조각 해시·ORB·perceptual hash의 원본 찾기.

설계·예측은 notes/개발일지.md "실험 19 설계". 결과는 E:/envs/c2pa-work/exp19/ (이미지마다 저장).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import fphash  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402
from exp05_tile_fingerprint import h64, mind, tiles  # noqa: E402
from exp11b_orb_squash import feats, inliers  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp19")
AREAS = [0.9, 0.8, 0.7, 0.6, 0.5]


def real_crop(im, area, seed):
    w, h = im.size
    s = area ** 0.5
    cw, ch = max(8, round(w * s)), max(8, round(h * s))
    rng = random.Random(seed)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return im.crop((x, y, x + cw, y + ch))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    n = len(items)
    T = [tiles(im) for _, _, im in items]
    H = [fphash.to_hex(fphash.lab_phash(im)) for _, _, im in items]
    R = [feats(im) for _, _, im in items]
    part = OUT / "rows.jsonl"
    done = {json.loads(l)[0] for l in part.read_text(encoding="utf-8").splitlines() if l} if part.exists() else set()
    for i, (name, kind, im) in enumerate(items):
        if i in done:
            continue
        out = []
        for a in AREAS:
            q = real_crop(im, a, i)
            hq = h64(q)
            pq = fphash.to_hex(fphash.lab_phash(q))
            fq = feats(q)
            out.append((i, kind, a, {"tile": [mind(hq, T[j]) for j in range(n)],
                                     "phash": [fphash.distance(pq, H[j]) for j in range(n)],
                                     "orb": [inliers(fq, R[j], "B") for j in range(n)]}))
        with part.open("a", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"{i + 1}/{n} {name}", flush=True)
    rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l]
    res = {"thresholds": {}, "tpr": {}}
    for v, better in [("tile", "low"), ("phash", "low"), ("orb", "high")]:
        if better == "low":
            neg = np.array([min(d for j, d in enumerate(r[3][v]) if j != r[0]) for r in rows])
            t = max([t for t in range(65) if np.mean(neg <= t) <= 0.01], default=-1)
            ok = lambda r: r[3][v][r[0]] <= t and r[3][v][r[0]] == min(r[3][v])  # noqa: E731
        else:
            neg = np.array([max(s for j, s in enumerate(r[3][v]) if j != r[0]) for r in rows])
            t = min([t for t in range(3000) if np.mean(neg >= t) <= 0.01], default=None)
            ok = lambda r: r[3][v][r[0]] >= t and r[3][v][r[0]] == max(r[3][v])  # noqa: E731
        res["thresholds"][v] = t
        for g in ["figure", "photo"]:
            res["tpr"][f"{v}/{g}"] = [float(np.mean([ok(r) for r in rows if r[1] == g and r[2] == a])) for a in AREAS]
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
