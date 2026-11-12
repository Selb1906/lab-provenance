"""내 비디오 워터마크 연구(core_AX v3 백본)의 이미지 이식판.

v3 백본(core_AX README, 2026-09-06 확정): 휘도 8x8 블록 DCT의 AC 계수 63개 전부에 ±1 반송파를 고르게 분산,
세기 alpha=15(계수당 alpha/sqrt(63)), 블록별 질감 적응 배수 w_j ∝ tex_j^0.75 (rms 정규화).
영상은 프레임을 모아 검출하지만 이미지는 한 장뿐이므로, 한 장 안의 모든 (블록, 계수)에서 상관을 구한다.
동기 탐색(회전·크롭 복원)은 아직 옮기지 않았다.

검출 통계: z = sum(C * s) / sqrt(sum(C^2)). 워터마크가 없고 반송파 s가 무작위 ±1이면 z ~ N(0, 1).
"""
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

B = 8
ALPHA = 15.0
GAMMA = 0.75


def _y(img):
    ycc = np.asarray(img.convert("YCbCr"), dtype=np.float64)
    return ycc


def _blocks(y):
    h, w = (y.shape[0] // B) * B, (y.shape[1] // B) * B
    v = y[:h, :w].reshape(h // B, B, w // B, B).transpose(0, 2, 1, 3)
    return v, h, w


def _carrier(key, nbr, nbc):
    rng = np.random.default_rng(key)
    s = rng.choice([-1.0, 1.0], size=(nbr, nbc, B, B))
    s[:, :, 0, 0] = 0.0  # DC 제외
    return s


def embed(img, key, alpha=ALPHA, gamma=GAMMA):
    ycc = _y(img)
    v, h, w = _blocks(ycc[:, :, 0])
    C = dctn(v, axes=(2, 3), norm="ortho")
    ac = C.copy()
    ac[:, :, 0, 0] = 0.0
    tex = np.sqrt((ac ** 2).sum(axis=(2, 3)) / 63.0)
    wj = np.power(np.maximum(tex, 1e-3), gamma) if gamma else np.ones_like(tex)
    wj = wj / np.sqrt(np.mean(wj ** 2))
    s = _carrier(key, *tex.shape)
    C = C + (alpha / np.sqrt(63.0)) * wj[:, :, None, None] * s
    v2 = idctn(C, axes=(2, 3), norm="ortho")
    y2 = ycc[:, :, 0].copy()
    y2[:h, :w] = v2.transpose(0, 2, 1, 3).reshape(h, w)
    ycc[:, :, 0] = y2
    out = Image.fromarray(np.clip(np.round(ycc), 0, 255).astype(np.uint8), "YCbCr").convert("RGB")
    return out


def score(img, key):
    v, _, _ = _blocks(_y(img)[:, :, 0])
    C = dctn(v, axes=(2, 3), norm="ortho")
    s = _carrier(key, C.shape[0], C.shape[1])
    num = (C * s).sum()
    den = np.sqrt((C[s != 0] ** 2).sum()) + 1e-12
    return float(num / den)
