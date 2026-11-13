"""실험 11: ORB 매칭의 검증 단계(글자 마스킹 + 아핀·크롭 가설 + 칸 수 점수)를 바꿔 잘라 쓴 그림 찾기.

설계·예측은 notes/개발일지.md "실험 11 설계". 결과는 E:/envs/c2pa-work/exp11/.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
from exp01_fingerprint import LEVELS, load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp11")
EAST = cv2.dnn.readNet("E:/envs/c2pa-work/models/east/east.pb")
ORB = cv2.ORB_create(nfeatures=1000)  # 실험 3과 같은 값
BF = cv2.BFMatcher(cv2.NORM_HAMMING)
GRID = 4
INSIDE_MIN = 0.9


def gray(im):
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)


def text_mask(g):
    """EAST로 글자 상자를 찾아 1(글자)로 표시한 마스크."""
    h, w = g.shape
    W = min(1280, max(32, (w // 32) * 32))
    H = min(1280, max(32, (h // 32) * 32))
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    blob = cv2.dnn.blobFromImage(rgb, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    EAST.setInput(blob)
    scores, geo = EAST.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    sc = scores[0, 0]
    m = np.zeros((h, w), np.uint8)
    rx, ry = w / W, h / H
    ys, xs = np.where(sc >= 0.5)
    for y, x in zip(ys, xs):
        d0, d1, d2, d3, ang = geo[0, :, y, x]
        cx, cy = x * 4.0, y * 4.0
        cos, sin = np.cos(ang), np.sin(ang)
        bw, bh = d1 + d3, d0 + d2
        ex = cx + cos * d1 + sin * d2
        ey = cy - sin * d1 + cos * d2
        x0, y0 = ex - bw, ey - bh
        px, py = 0.1 * bw, 0.1 * bh
        a0, b0 = int(max(0, (x0 - px) * rx)), int(max(0, (y0 - py) * ry))
        a1, b1 = int(min(w, (ex + px) * rx)), int(min(h, (ey + py) * ry))
        m[b0:b1, a0:a1] = 1
    return m


def feats(im, mask_text):
    g = gray(im)
    k, d = ORB.detectAndCompute(g, None)
    if mask_text and d is not None:
        m = text_mask(g)
        keep = [i for i, p in enumerate(k) if not m[min(int(p.pt[1]), g.shape[0] - 1), min(int(p.pt[0]), g.shape[1] - 1)]]
        k = [k[i] for i in keep]
        d = d[keep] if keep else None
    return k, d, g.shape


def good_matches(a, b):
    (ka, da, _), (kb, db, _) = a, b
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return None
    good = [m for m, n in (p for p in BF.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    return src, dst


def score_homography(q, r):
    gm = good_matches(q, r)
    if gm is None:
        return 0
    _, mask = cv2.findHomography(gm[0].reshape(-1, 1, 2), gm[1].reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def score_affine(q, r):
    """(inlier 수, 칸 수). 크롭 가설(질의 사각형이 참조 안에 들어감)을 통과 못 하면 (0, 0)."""
    gm = good_matches(q, r)
    if gm is None:
        return 0, 0
    src, dst = gm
    A, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)  # 5.0 = 실험 3과 같은 값
    if A is None:
        return 0, 0
    inl = mask.ravel().astype(bool)
    hq, wq = q[2]
    hr, wr = r[2]
    # 크롭 가설: 질의 위 격자점(21×21)을 옮겨 참조 안에 드는 비율
    gx, gy = np.meshgrid(np.linspace(0, wq, 21), np.linspace(0, hq, 21))
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 1) @ A.T
    inside = np.mean((pts[:, 0] >= -2) & (pts[:, 0] <= wr + 2) & (pts[:, 1] >= -2) & (pts[:, 1] <= hr + 2))
    if inside < INSIDE_MIN:
        return 0, 0
    p = src[inl]
    cells = {(min(int(y * GRID / hq), GRID - 1), min(int(x * GRID / wq), GRID - 1)) for x, y in p}
    return int(inl.sum()), len(cells)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    n = len(items)
    R0 = [feats(im, False) for _, _, im in items]
    R1 = [feats(im, True) for _, _, im in items]
    print("ref features done; mean kp", np.mean([len(r[0]) for r in R0]), "masked", np.mean([len(r[0]) for r in R1]), flush=True)
    rows = []  # (src, kind, attack, level, {variant: [score to each ref]})
    for i, (name, kind, im) in enumerate(items):
        for atk in ["resizedcrop", "rotation"]:
            for lv in LEVELS:
                q = attacks.apply(im, atk, lv, seed=i)
                q0, q1 = feats(q, False), feats(q, True)
                A, B, C, D = [], [], [], []
                for j in range(n):
                    A.append(score_homography(q0, R0[j]))
                    B.append(score_affine(q0, R0[j])[0])
                    c_inl, c_cells = score_affine(q1, R1[j])
                    C.append(c_inl)
                    D.append(c_cells)
                rows.append((i, kind, atk, lv, {"A": A, "B": B, "C": C, "D": D}))
        print(f"{i + 1}/{n} {name}", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows), encoding="utf-8")

    res = {"n_queries": len(rows), "thresholds": {}, "tpr": {}}
    for v in "ABCD":
        neg = np.array([max(s for j, s in enumerate(r[4][v]) if j != r[0]) for r in rows])
        t = min([t for t in range(0, 2000) if np.mean(neg >= t) <= 0.01], default=None)
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
