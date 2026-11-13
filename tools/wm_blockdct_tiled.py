"""블록 DCT 워터마크의 크롭 동기 탐색판 (실험 6).

wm_blockdct.py와 같은 삽입(alpha=15, 계수당 alpha/sqrt(63), 질감 적응 0.75)이지만 반송파를
P x P 블록(=128px) 주기로 반복한다. 검출은 후보 확대율마다 이미지를 원래 배율로 되돌린 뒤
128x128로 접어서(겹쳐 더해) 공간 영역 워터마크 무늬와 FFT 상호상관하고, 최대 봉우리를 점수로 쓴다.
"""
import numpy as np
from PIL import Image
from scipy.fft import dctn, fft2, idctn, ifft2

B = 8
P = 16            # 반송파 주기(블록)
T = B * P         # 128 px
ALPHA = 15.0
GAMMA = 0.75
SCALES = np.round(np.arange(1.00, 1.4501, 0.01), 2)


LOWFREQ = None    # None이면 AC 63개 전부(원래 v3). 정수 m이면 u+v <= m 인 저주파 AC만 사용


def _mask():
    u, v = np.meshgrid(np.arange(B), np.arange(B), indexing="ij")
    m = np.ones((B, B), bool) if LOWFREQ is None else (u + v <= LOWFREQ)
    m[0, 0] = False
    return m


def _tile_carrier(key):
    rng = np.random.default_rng(key)
    s = rng.choice([-1.0, 1.0], size=(P, P, B, B))
    s[:, :, ~_mask()] = 0.0
    return s


def _pattern(key):
    """128x128 공간 영역 무늬(질감 가중치 없이): 블록마다 idct(반송파)."""
    s = _tile_carrier(key)
    sp = idctn(s, axes=(2, 3), norm="ortho")
    return sp.transpose(0, 2, 1, 3).reshape(T, T)


def embed(img, key, alpha=ALPHA, gamma=GAMMA):
    ycc = np.asarray(img.convert("YCbCr"), dtype=np.float64).copy()
    y = ycc[:, :, 0]
    h, w = (y.shape[0] // B) * B, (y.shape[1] // B) * B
    v = y[:h, :w].reshape(h // B, B, w // B, B).transpose(0, 2, 1, 3)
    C = dctn(v, axes=(2, 3), norm="ortho")
    ac = C.copy()
    ac[:, :, 0, 0] = 0.0
    tex = np.sqrt((ac ** 2).sum(axis=(2, 3)) / 63.0)
    wj = np.power(np.maximum(tex, 1e-3), gamma)
    wj = wj / np.sqrt(np.mean(wj ** 2))
    base = _tile_carrier(key)
    nbr, nbc = tex.shape
    s = base[np.arange(nbr)[:, None] % P, np.arange(nbc)[None, :] % P]
    C = C + (alpha / np.sqrt(_mask().sum())) * wj[:, :, None, None] * s  # 블록당 신호 에너지 alpha^2 유지
    y2 = y.copy()
    y2[:h, :w] = idctn(C, axes=(2, 3), norm="ortho").transpose(0, 2, 1, 3).reshape(h, w)
    ycc[:, :, 0] = y2
    return Image.fromarray(np.clip(np.round(ycc), 0, 255).astype(np.uint8), "YCbCr").convert("RGB")


def _fold(y):
    h, w = (y.shape[0] // T) * T, (y.shape[1] // T) * T
    if h == 0 or w == 0:
        return None
    return y[:h, :w].reshape(h // T, T, w // T, T).sum(axis=(0, 2))


def score(img, key, scales=SCALES):
    """모든 후보 배율·위치에서 정규화 상호상관의 최대 봉우리(표준점수)와 그때의 배율."""
    pat = _pattern(key)
    Fp = np.conj(fft2(pat))
    y0 = np.asarray(img.convert("YCbCr"), dtype=np.float64)[:, :, 0]
    best, best_k = -np.inf, None
    im = Image.fromarray(np.clip(y0, 0, 255).astype(np.uint8))
    for k in scales:
        if k == 1.0:
            y = y0
        else:
            w, h = im.size
            y = np.asarray(im.resize((max(T, round(w / k)), max(T, round(h / k))), Image.BILINEAR), dtype=np.float64)
        f = _fold(y - y.mean())
        if f is None:
            continue
        c = np.real(ifft2(fft2(f) * Fp))
        z = (c.max() - c.mean()) / (c.std() + 1e-12)
        if z > best:
            best, best_k = z, float(k)
    return float(best), best_k
