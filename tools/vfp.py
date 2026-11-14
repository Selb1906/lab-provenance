"""영상 fingerprint — Meta ThreatExchange vPDQ 방식(vpdq/README.md, 조사 06 3.2)을 README 예시값으로.

프레임을 초당 1장 뽑아 PDQ 256비트 + 품질을 만든다. 매칭: 질의 프레임이 비교 영상의 어느 프레임과 거리 D 이내면 '맞음',
품질 F 미만 프레임은 뺀다. 비교측 매칭 비율 ≥ P_C 그리고 질의측 ≥ P_Q면 같은 영상.
"""
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pdqhash
from PIL import Image

D, F, P_C, P_Q = 31, 50, 0.80, 0.0  # README 예시값


def frame_hashes(path, fps=1, hflip=False):
    """(해시 배열 (n,256) uint8, 품질 배열 (n,)). hflip=True면 좌우 반전한 프레임으로(반전해 올린 영상 대비)."""
    with tempfile.TemporaryDirectory(dir="E:/envs/tmp") as d:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps}" + (",hflip" if hflip else ""),
                        str(Path(d) / "f%04d.png")], check=True)
        hs, qs = [], []
        for f in sorted(Path(d).glob("f*.png")):
            h, q = pdqhash.compute(np.asarray(Image.open(f).convert("RGB")))
            hs.append(np.asarray(h, np.uint8))
            qs.append(q)
    return np.array(hs), np.array(qs)


def match(q, r):
    """q, r: frame_hashes 결과 → (질의측 비율, 비교측 비율)."""
    (hq, qq), (hr, qr) = q, r
    hq, hr = hq[qq >= F], hr[qr >= F]
    if len(hq) == 0 or len(hr) == 0:
        return 0.0, 0.0
    dist = (hq[:, None, :] != hr[None, :, :]).sum(axis=2)  # (nq, nr)
    close = dist <= D
    return float(close.any(axis=1).mean()), float(close.any(axis=0).mean())


def same(q, r):
    pq, pc = match(q, r)
    return pc >= P_C and pq >= P_Q
