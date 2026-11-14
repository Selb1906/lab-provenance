"""실험 23: 프레임마다 block DCT(같은 반송파)를 심은 영상을 재인코딩·축소·잘라 올린 뒤, 프레임 하나 vs 프레임 누적 z.

설계·예측은 notes/개발일지.md "실험 23 설계". 결과는 E:/envs/c2pa-work/exp23/summary.json.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy.fft import dctn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import wm_blockdct as W  # noqa: E402

SRC = Path("E:/envs/c2pa-work/exp21/src")
OUT = Path("E:/envs/c2pa-work/exp23")
FF = imageio_ffmpeg.get_ffmpeg_exe()
KEY = W.derive_key(int(Path("E:/envs/c2pa-site-keys/wm_key.txt").read_text().strip()), "VIDEO-TEST")
Z_T = 2.326


def frames_of(path, size=None):
    with tempfile.TemporaryDirectory(dir="E:/envs/tmp") as d:
        vf = ["-vf", f"scale={size[0]}:{size[1]}"] if size else []
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(path), *vf, str(Path(d) / "f%04d.png")], check=True)
        return [np.asarray(Image.open(f).convert("RGB")) for f in sorted(Path(d).glob("f*.png"))]


def encode(frames, out, crf, fps=30):
    with tempfile.TemporaryDirectory(dir="E:/envs/tmp") as d:
        for k, f in enumerate(frames):
            Image.fromarray(f).save(Path(d) / f"f{k:04d}.png")
        subprocess.run([FF, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(d) / "f%04d.png"),
                        "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)], check=True)


def corr_parts(frame, key):
    """프레임 하나의 Σ(C·s), Σ C² (반송파가 0이 아닌 계수만) — 누적 z를 만들려고 따로 돌려준다."""
    v, _, _ = W._blocks(W._y(Image.fromarray(frame))[:, :, 0])
    C = dctn(v, axes=(2, 3), norm="ortho")
    s = W._carrier(key, C.shape[0], C.shape[1])
    return float((C * s).sum()), float((C[s != 0] ** 2).sum())


def zs(frames, key):
    parts = [corr_parts(f, key) for f in frames]
    single = [n / np.sqrt(d) for n, d in parts]
    total = sum(n for n, _ in parts) / np.sqrt(sum(d for _, d in parts))
    return float(np.median(single)), float(total)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for k in range(10):
        src = SRC / f"v{k}.mp4"
        fr = frames_of(src)
        w, h = fr[0].shape[1], fr[0].shape[0]
        marked = [np.asarray(W.embed(Image.fromarray(f), KEY)) for f in fr]
        base = OUT / f"m{k}.mp4"
        encode(marked, base, 18)
        row = {"v": k, "frames": len(fr)}
        for name, crf, scale, cut in [("crf18", 18, None, None), ("crf28", 28, None, None), ("crf35", 35, None, None),
                                      ("480p", 23, (854, 480), None), ("sub2s", 23, None, (3, 2))]:
            q = OUT / f"q{k}_{name}.mp4"
            vf = ["-vf", f"scale={scale[0]}:{scale[1]}"] if scale else []
            cutargs = ["-ss", str(cut[0]), "-t", str(cut[1])] if cut else []
            subprocess.run([FF, "-y", "-loglevel", "error", *cutargs, "-i", str(base), *vf, "-c:v", "libx264", "-crf", str(crf),
                            "-pix_fmt", "yuv420p", str(q)], check=True)
            qf = frames_of(q, size=(w, h) if scale else None)  # 축소본은 원래 크기로 다시 늘려 블록 격자를 맞춤
            z1, zt = zs(qf, KEY)
            z1n, ztn = zs(qf, KEY + 1)  # 음성 대조(틀린 key)
            row[name] = {"z_single_median": z1, "z_accum": zt, "n": len(qf), "neg_single": z1n, "neg_accum": ztn}
            q.unlink()
        res.append(row)
        print(k, {n: (round(row[n]["z_single_median"], 1), round(row[n]["z_accum"], 1)) for n in ["crf18", "crf28", "crf35", "480p", "sub2s"]}, flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
