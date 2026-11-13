"""원본에 맞춰 되돌린 뒤 워터마크 읽기 (실험 11b·12에서 옮김).

1) candidate(): 질의를 미리 눌러 본 여러 판(늘어난 크롭 대응)과 각 원본의 ORB 짝을 맞춰 아핀 inlier가 가장 많은 원본을 고른다.
2) register(): 고른 원본에 SIFT로 정합해 질의를 원본 좌표로 되돌린다(원본과 같은 크기, 유효 영역 마스크).
3) bdct_z_valid(): 유효 영역이 블록 전체를 덮는 블록만으로 블록 DCT z를 계산(실험 12: 다른 키 z가 N(0,1)을 따름 확인).
판정은 워터마크 검정(z > 2.326, 오탐 1%)이 맡고, 1)은 어느 원본을 시험할지 고르는 데만 쓴다.
"""
import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn

import wm_blockdct as W

ORB = cv2.ORB_create(nfeatures=1000)  # 실험 3·11과 같은 값
BF_H = cv2.BFMatcher(cv2.NORM_HAMMING)
SIFT = cv2.SIFT_create()  # 조사 10: 기본값
BF_L2 = cv2.BFMatcher(cv2.NORM_L2)
Z_T = 2.326


def _gray(im):
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)


def squash_factors(w, h):
    """1부터 가로세로비까지 등비 6단계(실험 12). 정사각 크롭을 원래 가로세로로 늘린 질의를 되돌려 보기 위한 후보."""
    r = max(w / h, h / w)
    return sorted({1.0, *np.round(np.geomspace(1.0, r, 6), 3)})


def _squashed(q):
    w, h = q.size
    for f in squash_factors(w, h):
        S = np.diag([1 / f, 1.0, 1.0]) if w >= h else np.diag([1.0, 1 / f, 1.0])
        size = (max(8, round(w * S[0, 0])), max(8, round(h * S[1, 1])))
        yield (q if f == 1.0 else q.resize(size, Image.BILINEAR)), S


def _matches(kq, dq, ko, do, bf):
    if dq is None or do is None or len(kq) < 4 or len(ko) < 4:
        return None
    good = [m for m, n in (p for p in bf.knnMatch(dq, do, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None
    return (np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2),
            np.float32([ko[m.trainIdx].pt for m in good]).reshape(-1, 1, 2))


def orb_features(im):
    return ORB.detectAndCompute(_gray(im), None)


def candidate(q, refs):
    """refs: [(이름, ORB 특징)] → (가장 inlier가 많은 이름, inlier 수)."""
    qs = [(ORB.detectAndCompute(_gray(v), None)) for v, _ in _squashed(q)]
    best = (None, 0)
    for name, (ko, do) in refs:
        for kq, dq in qs:
            m = _matches(kq, dq, ko, do, BF_H)
            if m is None:
                continue
            _, inl = cv2.estimateAffine2D(m[0], m[1], method=cv2.RANSAC, ransacReprojThreshold=5.0)
            n = int(inl.sum()) if inl is not None else 0
            if n > best[1]:
                best = (name, n)
    return best


def register(q, o):
    """질의 q를 원본 o 좌표로 되돌린다 → (되돌린 이미지, 유효 마스크, 변환, inlier 수) 또는 None."""
    go = _gray(o)
    ko, do = SIFT.detectAndCompute(go, None)
    best = None
    for qs, S in _squashed(q):
        kq, dq = SIFT.detectAndCompute(_gray(qs), None)
        m = _matches(kq, dq, ko, do, BF_L2)
        if m is None:
            continue
        A, inl = cv2.estimateAffine2D(m[0], m[1], method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if A is None:
            continue
        n = int(inl.sum())
        if best is None or n > best[1]:
            best = (np.vstack([A, [0, 0, 1]]) @ S, n)
    if best is None:
        return None
    M, n = best
    H, Wd = go.shape
    rgb = np.asarray(q.convert("RGB"))
    back = cv2.warpPerspective(rgb, M, (Wd, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(np.full(rgb.shape[:2], 255, np.uint8), M, (Wd, H), flags=cv2.INTER_NEAREST) > 0
    return Image.fromarray(back), valid, M, n


def bdct_z_valid(img, valid, key):
    v, h, w = W._blocks(W._y(img)[:, :, 0])
    C = dctn(v, axes=(2, 3), norm="ortho")
    s = W._carrier(key, C.shape[0], C.shape[1])
    vb = valid[:h, :w].reshape(h // 8, 8, w // 8, 8).all(axis=(1, 3))
    Cs, ss = C[vb], s[vb]
    if Cs.size == 0:
        return 0.0
    return float((Cs * ss).sum() / (np.sqrt((Cs[ss != 0] ** 2).sum()) + 1e-12))
