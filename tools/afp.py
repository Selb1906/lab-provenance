"""음원 fingerprint — Philips 방식(Haitsma & Kalker, "A Highly Robust Audio Fingerprinting System", ISMIR 2002) 그대로.

논문 설정(2절): 5kHz로 다시 샘플링, 프레임 0.37초(2048표본)·겹침 31/32(간격 64표본, 11.6ms),
300~2000Hz를 로그 간격 33대역으로 나눈 에너지 E(n, m)에서
  F(n, m) = 1  if (E(n,m) − E(n,m+1)) − (E(n−1,m) − E(n−1,m+1)) > 0  else 0     (m = 0..31)
로 프레임마다 32비트를 만든다. 두 음원은 비트 오류율(BER)로 비교하고, 논문은 BER < 0.35를 같은 음원으로 본다.
"""
import numpy as np
from scipy.signal import resample_poly

FS = 5000
FRAME = 2048
HOP = 64
F_LO, F_HI, NB = 300.0, 2000.0, 33
BER_T = 0.35


def _bands():
    edges = np.geomspace(F_LO, F_HI, NB + 1)
    freqs = np.fft.rfftfreq(FRAME, 1 / FS)
    return [(freqs >= edges[b]) & (freqs < edges[b + 1]) for b in range(NB)]


_B = _bands()
_WIN = np.hanning(FRAME)


def fingerprint(x, sr):
    """x: 1차원 음원(실수), sr: 표본율 → (프레임 수, 32) uint8 비트."""
    x = np.asarray(x, np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    g = np.gcd(int(sr), FS)
    y = resample_poly(x, FS // g, int(sr) // g)
    n = 1 + (len(y) - FRAME) // HOP
    if n < 2:
        return np.zeros((0, 32), np.uint8)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * _WIN, axis=1)) ** 2
    E = np.stack([spec[:, m].sum(axis=1) for m in _B], axis=1)  # (n, 33)
    d = E[:, :-1] - E[:, 1:]                                      # (n, 32)
    return ((d[1:] - d[:-1]) > 0).astype(np.uint8)                # (n-1, 32)


def ber(a, b, max_shift=8):
    """두 fingerprint의 비트 오류율. 재압축으로 앞뒤가 몇 프레임 밀릴 수 있어 ±max_shift 프레임 중 최소를 쓴다."""
    best = 1.0
    for s in range(-max_shift, max_shift + 1):
        x, y = (a[s:], b) if s >= 0 else (a, b[-s:])
        m = min(len(x), len(y))
        if m < 64:
            continue
        best = min(best, float(np.mean(x[:m] != y[:m])))
    return best
