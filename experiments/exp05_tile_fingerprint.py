"""실험 5: 조각 단위 lab-phash로 크롭 빈틈 메우기. 설계·예측은 notes/개발일지.md "실험 5 설계"."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
import fphash  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp05")
AREAS = [0.5, 0.6, 0.7, 0.8, 0.9]
GRID = 5
LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]


def h64(im):
    return int(fphash.to_hex(fphash.lab_phash(im)), 16)


def tiles(im):
    w, h = im.size
    out = [h64(im)]
    for s in AREAS:
        side = max(1, min(int(round((s * w * h) ** 0.5)), w, h))
        for gy in range(GRID):
            for gx in range(GRID):
                left = round(gx * (w - side) / (GRID - 1))
                top = round(gy * (h - side) / (GRID - 1))
                out.append(h64(im.crop((left, top, left + side, top + side))))
    return np.array(out, dtype=np.uint64)


def popcount(x):
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.int64)
    for _ in range(64):
        c += (x & np.uint64(1)).astype(np.int64)
        x >>= np.uint64(1)
    return c


def mind(q, arr):
    return int(popcount(np.bitwise_xor(arr, np.uint64(q))).min())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    T = [tiles(im) for _, _, im in items]
    print("tiles per image:", len(T[0]), flush=True)
    kinds = [k for _, k, _ in items]
    # 오탐 분포: 각 원본 전체 해시가 다른 이미지의 (전체+조각) 중 가장 가까운 거리 + 공격 질의도 다른 이미지와 비교
    imp, gen = [], []
    for i, (name, kind, im) in enumerate(items):
        qs = [(None, None, h64(im))]
        for lv in LEVELS:
            qs.append(("resizedcrop", lv, h64(attacks.apply(im, "resizedcrop", lv, seed=i))))
        for atk, lv, q in qs:
            for j in range(len(items)):
                if j != i:
                    imp.append((kind, kinds[j], mind(q, T[j])))
            if atk:
                gen.append((kind, lv, mind(q, T[i]), mind(q, T[i][:1])))
        print(f"{i + 1}/{len(items)} {name}", flush=True)
    d = np.array([x[2] for x in imp])
    t = max(t for t in range(65) if np.mean(d <= t) <= 0.01)
    res = {"tiles_per_image": int(len(T[0])), "t_tile": t, "fpr": float(np.mean(d <= t)), "n_impostor": len(d), "tpr": {}}
    for g in ["figure", "photo"]:
        dg = np.array([x[2] for x in imp if x[0] == g and x[1] == g])
        res[f"fpr_{g}_vs_{g}"] = float(np.mean(dg <= t))
        for lv in LEVELS:
            rows = [x for x in gen if x[0] == g and x[1] == lv]
            res["tpr"][f"{g}/tiles/{lv}"] = float(np.mean([r[2] <= t for r in rows]))
            res["tpr"][f"{g}/whole_t15/{lv}"] = float(np.mean([r[3] <= 15 for r in rows]))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
