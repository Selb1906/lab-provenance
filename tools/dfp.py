"""문서 텍스트 fingerprint — Broder(1997) MinHash, 연속 5단어 shingle, 해시 128개.

유사도(Jaccard 추정)와 포함도(질의 shingle 중 원본에도 있는 비율 추정)를 함께 계산한다.
포함도는 MinHash 서명만으로는 정확히 못 구하므로, 원장에는 서명과 함께 shingle 해시 집합의 작은 표본
(값이 작은 순 512개 — bottom-k 표본, Broder 1997의 '최솟값 표본' 방식)을 저장해 추정한다.
"""
import hashlib
import re

import numpy as np

K_SHINGLE = 5
N_PERM = 128
BOTTOM_K = 512
_P = (1 << 61) - 1
_rng = np.random.default_rng(20261114)  # 해시 함수 계수 고정(원장과 조회가 같은 함수를 써야 함)
_A = _rng.integers(1, _P, N_PERM, dtype=np.uint64)
_B = _rng.integers(0, _P, N_PERM, dtype=np.uint64)


def words(text):
    return re.findall(r"[0-9a-z가-힣]+", text.lower())


def shingles(text):
    w = words(text)
    # SHA-256 앞 8바이트(브라우저 WebCrypto와 같게 — 처음 판은 blake2b였음, 11-15 변경). >>3 으로 61비트 소수 아래로
    return {int.from_bytes(hashlib.sha256(" ".join(w[i:i + K_SHINGLE]).encode()).digest()[:8], "big") >> 3
            for i in range(max(0, len(w) - K_SHINGLE + 1))}


def signature(sh):
    x = np.fromiter(sh, dtype=np.uint64)
    if len(x) == 0:
        return np.full(N_PERM, _P, np.uint64)
    # (a·x + b) mod p 를 해시마다 계산해 최솟값. 큰 수 곱 넘침을 피하려고 파이썬 정수로
    xs = [int(v) for v in x]
    return np.array([min((int(a) * v + int(b)) % _P for v in xs) for a, b in zip(_A, _B)], dtype=np.uint64)


def bottom_k(sh, k=BOTTOM_K):
    return sorted(sh)[:k]


def jaccard(sa, sb):
    return float(np.mean(sa == sb))


def containment(q_sh, ref_bottom):
    """질의 shingle 중 원본에도 있는 비율의 추정: 원본 bottom-k 표본의 범위(최댓값) 안에 드는 질의 shingle만 보고 센다."""
    if not ref_bottom or not q_sh:
        return 0.0
    hi = ref_bottom[-1]
    q_in = [v for v in q_sh if v <= hi]
    if not q_in:
        return 0.0
    ref = set(ref_bottom)
    return sum(v in ref for v in q_in) / len(q_in)


def docx_text(path):
    """docx 본문: word/document.xml의 <w:t> 글자를 문단(<w:p>)마다 줄바꿈으로 잇는다(브라우저 dfp.js docxText와 같은 규칙)."""
    import html
    import zipfile
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    paras = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
    return "\n".join(html.unescape("".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", p, flags=re.S))) for p in paras)
