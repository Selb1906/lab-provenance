"""실험 12: 의심 이미지를 원본에 SIFT로 정합해 되돌린 뒤 워터마크(블록 DCT·TrustMark) 검출.

방법은 notes/research/10_다음실험_준비.md 1절을 그대로 따른다. 설계·예측은 notes/개발일지.md "실험 12 설계".
결과는 E:/envs/c2pa-work/exp12/.
"""
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
import attacks  # noqa: E402
import wm_blockdct as W  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp12")
KEY = int(Path("E:/envs/c2pa-site-keys/wm_key.txt").read_text().strip())
LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
Z_T = 2.326
cv2.setRNGSeed(0)
SIFT = cv2.SIFT_create()
BF = cv2.BFMatcher(cv2.NORM_L2)


def true_map(kind, rel, seed, w, h):
    """질의 좌표 → 원본 좌표 (참 변환, 3x3)."""
    s = attacks.absolute(kind, rel)
    if kind == "resizedcrop":
        top, left, side = attacks._square_params(w, h, s, random.Random(seed))
        sx, sy = side / w, side / h
        # PIL resize(BILINEAR)의 화소 중심 규약: 원본 x = left + (x + 0.5) * sx - 0.5
        return np.array([[sx, 0, left + 0.5 * sx - 0.5], [0, sy, top + 0.5 * sy - 0.5], [0, 0, 1]])
    th = math.radians(s)
    cx, cy = w / 2, h / 2
    c, si = math.cos(th), math.sin(th)
    # PIL rotate(각도)는 화면에서 반시계 회전. 되돌리기: o = c + R(-θ)(q - c) (y가 아래로 커지는 좌표계)
    R = np.array([[c, -si], [si, c]])
    t = np.array([cx, cy]) - R @ np.array([cx, cy])
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]])


# SIFT는 회전·균일 배율에만 불변이라, 정사각 크롭을 가로로 늘린 질의(배율 비 최대 ≈ 원본 가로세로비)는 짝이 안 맞는다
# (사전 시험: 1500×779 그림의 면적 0.8 크롭에서 inlier 6, 모서리 오차 2,719px). 그래서 질의의 긴 축을 몇 배로 미리 눌러
# 보고 inlier가 가장 많은 것을 고른다. 후보는 1부터 원본 가로세로비까지 6단계(이 공격이 만들 수 있는 배율 비의 범위).
def _squash_factors(w, h):
    r = max(w / h, h / w)
    return sorted({1.0, *np.round(np.geomspace(1.0, r, 6), 3)})


def register(q_pil, o_pil, model):
    best = None
    w, h = q_pil.size
    for f in _squash_factors(w, h):
        S = np.diag([1 / f, 1.0, 1.0]) if w >= h else np.diag([1.0, 1 / f, 1.0])
        qs = q_pil.resize((max(8, round(w * S[0, 0])), max(8, round(h * S[1, 1]))), Image.BILINEAR)
        r = _register_once(qs, o_pil, model, q_pil, S)
        if r is not None and (best is None or r[3] > best[3]):
            best = r
    return best


def _register_once(qs_pil, o_pil, model, q_pil, S):
    q = cv2.cvtColor(np.asarray(qs_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    o = cv2.cvtColor(np.asarray(o_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    kq, dq = SIFT.detectAndCompute(q, None)
    ko, do = SIFT.detectAndCompute(o, None)
    if dq is None or do is None or len(kq) < 4 or len(ko) < 4:
        return None
    good = [m for m, n in (p for p in BF.knnMatch(dq, do, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None
    src = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ko[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    if model == "affine":
        A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if A is None:
            return None
        M = np.vstack([A, [0, 0, 1]])
    else:
        M, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if M is None:
            return None
    M = M @ S  # 눌러 본 좌표 → 원래 질의 좌표로 합성: 원래 질의 → 원본
    H, Wd = o.shape
    rgb = np.asarray(q_pil.convert("RGB"))
    back = cv2.warpPerspective(rgb, M, (Wd, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(np.full(rgb.shape[:2], 255, np.uint8), M, (Wd, H), flags=cv2.INTER_NEAREST) > 0
    return Image.fromarray(back), valid, M, int(inl.sum()), len(kq), len(ko)


def corner_err(M, T, w, h):
    pts = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], float).T
    a, b = M @ pts, T @ pts
    a, b = a[:2] / a[2], b[:2] / b[2]
    return float(np.mean(np.linalg.norm(a - b, axis=0)))


def bdct_z_valid(img, valid, key):
    """유효 영역이 블록 전체를 덮는 블록만으로 z = Σ(C·s)/√Σ(C²)."""
    v, h, w = W._blocks(W._y(img)[:, :, 0])
    C = dctn(v, axes=(2, 3), norm="ortho")
    s = W._carrier(key, C.shape[0], C.shape[1])
    vb = valid[:h, :w].reshape(h // 8, 8, w // 8, 8).all(axis=(1, 3))
    Cs, ss = C[vb], s[vb]
    den = np.sqrt((Cs[ss != 0] ** 2).sum()) + 1e-12
    return float((Cs * ss).sum() / den), int(vb.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from trustmark import TrustMark
    tm = TrustMark(verbose=False, model_type="Q", loadRemover=False)
    figs = json.loads((ROOT / "docs" / "assets" / "figures.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(1)
    rows = []
    for i, f in enumerate(figs):
        o = Image.open(ROOT / "docs" / f["file"]).convert("RGB")
        w, h = o.size
        for kind in ["resizedcrop", "rotation"]:
            for lv in LEVELS:
                q = attacks.apply(o, kind, lv, seed=i)
                T = true_map(kind, lv, i, w, h)
                s1, ok1, _ = tm.decode(q, MODE="text")
                row = {"fig": f["file"], "kind": kind, "level": lv, "wid": f["watermark_id"],
                       "direct_bdct": W.score(q, KEY), "direct_tm": bool(ok1 and s1 == f["watermark_id"])}
                for model in ["affine", "homography"]:
                    r = register(q, o, model)
                    if r is None:
                        row[model] = None
                        continue
                    back, valid, M, ninl, nkq, nko = r
                    z, nb = bdct_z_valid(back, valid, KEY)
                    z0, _ = bdct_z_valid(back, valid, int(rng.integers(1, 2**62)))
                    s2, ok2, _ = tm.decode(back, MODE="text")
                    row[model] = {"corner_err": corner_err(M, T, w, h), "inliers": ninl, "kp_q": nkq, "kp_o": nko,
                                  "bdct_z": z, "bdct_z_wrongkey": z0, "valid_blocks": nb,
                                  "tm": bool(ok2 and s2 == f["watermark_id"]), "tm_text": s2 if ok2 else None}
                rows.append(row)
                a = row["affine"] or {}
                print(f'{f["file"].split("/")[-1]} {kind} {lv}: err={a.get("corner_err", -1):.2f} z={a.get("bdct_z", 0):.1f} '
                      f'z0={a.get("bdct_z_wrongkey", 0):.1f} tm={a.get("tm")} | direct z={row["direct_bdct"]:.1f} tm={row["direct_tm"]}', flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
