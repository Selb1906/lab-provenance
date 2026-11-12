"""브라우저(verify.js)와 똑같이 계산되는 지각 해시 (lab-phash, 64비트).

imagehash.phash는 PIL의 LANCZOS 축소에 의존해 브라우저에서 같은 값을 내기 어렵다.
그래서 두 환경에서 정수 연산으로 똑같이 재현되는 규격을 정했다:
1) 회색조 Y = (299R + 587G + 114B) / 1000
2) 32x32 구역 평균 축소 (구역 경계는 floor(i*W/32))
3) 2차원 DCT-II (정규화 없음), 왼쪽 위 8x8
4) 64개 값의 중앙값보다 크면 1
"""
import numpy as np
from PIL import Image

N = 32


def _shrink(gray):
    h, w = gray.shape
    out = np.zeros((N, N))
    for i in range(N):
        r0, r1 = (i * h) // N, max((i + 1) * h // N, (i * h) // N + 1)
        for j in range(N):
            c0, c1 = (j * w) // N, max((j + 1) * w // N, (j * w) // N + 1)
            out[i, j] = gray[r0:r1, c0:c1].mean()
    return out


def _dct_matrix():
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    return np.cos(np.pi * (2 * n + 1) * k / (2 * N))


def lab_phash(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    gray = (299 * a[:, :, 0] + 587 * a[:, :, 1] + 114 * a[:, :, 2]) / 1000.0
    s = _shrink(gray)
    D = _dct_matrix()
    c = (D @ s @ D.T)[:8, :8].ravel()
    med = np.median(c)
    bits = c > med
    return "".join("1" if b else "0" for b in bits)


def to_hex(bits):
    return f"{int(bits, 2):016x}"


def distance(h1, h2):
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


if __name__ == "__main__":
    import sys
    print(to_hex(lab_phash(Image.open(sys.argv[1]))))
