"""실험 3: 크롭·회전에 버티는 핑거프린트 — PDQ와 ORB 특징점 매칭.

설계와 예측은 notes/개발일지.md "실험 3 설계". 결과는 E:/envs/c2pa-work/exp03/.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pdqhash

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp03")
LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
ATTACKS = ["rotation", "resizedcrop", "erasing", "compression"]
ORB = cv2.ORB_create(nfeatures=1000)
BF = cv2.BFMatcher(cv2.NORM_HAMMING)


def pdq(im):
    v, _ = pdqhash.compute(np.asarray(im.convert("RGB")))
    return np.asarray(v, dtype=np.uint8)


def orb(im):
    g = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return ORB.detectAndCompute(g, None)


def orb_score(a, b):
    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return 0
    good = [m for m, n in (p for p in BF.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return len(good)
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    P = [pdq(im) for _, _, im in items]
    O = [orb(im) for _, _, im in items]
    kinds = [k for _, k, _ in items]
    imp = {"pdq": [], "orb": [], "kind": []}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            imp["pdq"].append(int((P[i] != P[j]).sum()))
            imp["orb"].append(orb_score(O[i], O[j]))
            imp["kind"].append((kinds[i], kinds[j]))
    gen = []
    for idx, (name, kind, im) in enumerate(items):
        for atk in ATTACKS:
            for lv in LEVELS:
                a = attacks.apply(im, atk, lv, seed=idx)
                gen.append((idx, kind, atk, lv, int((P[idx] != pdq(a)).sum()), orb_score(orb(a), O[idx])))
        print(f"{idx + 1}/{len(items)} {name}", flush=True)
    d_pdq = np.array(imp["pdq"])
    d_orb = np.array(imp["orb"])
    t_pdq = max(t for t in range(257) if np.mean(d_pdq <= t) <= 0.01)       # 거리 <= t 이면 같다
    t_orb = min(t for t in range(0, 2000) if np.mean(d_orb >= t) <= 0.01)   # inlier >= t 이면 같다
    summary = {"t_pdq": t_pdq, "t_orb": t_orb, "fpr_pdq": float(np.mean(d_pdq <= t_pdq)),
               "fpr_orb": float(np.mean(d_orb >= t_orb)), "tpr": {}}
    for g in ["figure", "photo"]:
        for atk in ATTACKS:
            for lv in LEVELS:
                rows = [r for r in gen if r[1] == g and r[2] == atk and r[3] == lv]
                summary["tpr"][f"{g}/pdq/{atk}/{lv}"] = float(np.mean([r[4] <= t_pdq for r in rows]))
                summary["tpr"][f"{g}/orb/{atk}/{lv}"] = float(np.mean([r[5] >= t_orb for r in rows]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "raw.json").write_text(json.dumps({"imp": imp, "gen": gen}), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "tpr"}))


if __name__ == "__main__":
    main()
