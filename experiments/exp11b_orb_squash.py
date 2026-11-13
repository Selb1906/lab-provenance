"""실험 11b: 실험 11에 '질의 미리 눌러 보기'(실험 12와 같은 6단계)를 넣어, 늘어난 크롭 질의에서도 원본을 찾는지 잰다.

크롭 가설 조건은 회전 질의를 떨어뜨렸으므로 뺀다. 글자 마스킹은 효과가 작아 뺀다. 변형: A 호모그래피, B 아핀(둘 다 inlier 수).
설계는 notes/개발일지.md "실험 11 결과". 결과는 E:/envs/c2pa-work/exp11b/ (이미지마다 중간 저장).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
from exp01_fingerprint import LEVELS, load_set  # noqa: E402
from exp11_orb_verify import ORB, good_matches  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp11b")


def feats(im):
    g = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    k, d = ORB.detectAndCompute(g, None)
    return k, d, g.shape


def squash_versions(q):
    w, h = q.size
    r = max(w / h, h / w)
    out = []
    for f in sorted({1.0, *np.round(np.geomspace(1.0, r, 6), 3)}):
        size = (max(8, round(w / f)), h) if w >= h else (w, max(8, round(h / f)))
        out.append(feats(q.resize(size, Image.BILINEAR)) if f != 1.0 else feats(q))
    return out


def inliers(q, r, model):
    gm = good_matches(q, r)
    if gm is None:
        return 0
    src, dst = gm
    if model == "A":
        _, m = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    else:
        _, m = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return int(m.sum()) if m is not None else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    n = len(items)
    R = [feats(im) for _, _, im in items]
    part = OUT / "rows.jsonl"
    done = set()
    if part.exists():
        done = {json.loads(l)[0] for l in part.read_text(encoding="utf-8").splitlines() if l}
    for i, (name, kind, im) in enumerate(items):
        if i in done:
            continue
        rows = []
        for atk in ["resizedcrop", "rotation"]:
            for lv in LEVELS:
                Q = squash_versions(attacks.apply(im, atk, lv, seed=i))
                sc = {v: [max(inliers(q, R[j], v) for q in Q) for j in range(n)] for v in "AB"}
                rows.append((i, kind, atk, lv, sc))
        with part.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{i + 1}/{n} {name}", flush=True)

    rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l]
    res = {"n_queries": len(rows), "thresholds": {}, "tpr": {}}
    for v in "AB":
        neg = np.array([max(s for j, s in enumerate(r[4][v]) if j != r[0]) for r in rows])
        t = min([t for t in range(0, 3000) if np.mean(neg >= t) <= 0.01], default=None)
        res["thresholds"][v] = {"t": t, "fpr": float(np.mean(neg >= t)), "neg_max": int(neg.max())}
        for g in ["figure", "photo"]:
            for atk in ["resizedcrop", "rotation"]:
                for lv in LEVELS:
                    rs = [r for r in rows if r[1] == g and r[2] == atk and r[3] == lv]
                    ok = [r[4][v][r[0]] >= t and r[4][v][r[0]] == max(r[4][v]) for r in rs]
                    res["tpr"][f"{v}/{g}/{atk}/{lv}"] = float(np.mean(ok))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["thresholds"], indent=1))


if __name__ == "__main__":
    main()
