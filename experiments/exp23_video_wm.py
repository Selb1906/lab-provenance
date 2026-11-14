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


# 첫 실행(11-15 03:09)은 720p 프레임 240장을 원본·삽입본·공격본마다 리스트로 메모리에 올려(영상 하나에 1GB 넘음)
# 시스템 메모리 부족으로 자동 중지됨 → 프레임은 디스크 PNG로 두고 한 장씩 읽어 처리한다.
def extract(path, d, size=None):
    vf = ["-vf", f"scale={size[0]}:{size[1]}"] if size else []
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(path), *vf, str(Path(d) / "f%04d.png")], check=True)
    return sorted(Path(d).glob("f*.png"))


def encode_dir(d, out, crf, fps=30):
    subprocess.run([FF, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(d) / "f%04d.png"),
                    "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)], check=True)


def corr_parts(frame, key):
    """프레임 하나의 Σ(C·s), Σ C² (반송파가 0이 아닌 계수만) — 누적 z를 만들려고 따로 돌려준다."""
    v, _, _ = W._blocks(W._y(Image.fromarray(frame))[:, :, 0])
    C = dctn(v, axes=(2, 3), norm="ortho")
    s = W._carrier(key, C.shape[0], C.shape[1])
    return float((C * s).sum()), float((C[s != 0] ** 2).sum())


def zs(paths, keys):
    """프레임 PNG 경로들 → key마다 (프레임 하나 z 중앙값, 누적 z). 한 장씩 읽는다."""
    acc = {k: [0.0, 0.0, []] for k in keys}
    for p in paths:
        f = np.asarray(Image.open(p).convert("RGB"))
        for k in keys:
            n, d = corr_parts(f, k)
            acc[k][0] += n
            acc[k][1] += d
            acc[k][2].append(n / np.sqrt(d))
    return {k: (float(np.median(v[2])), float(v[0] / np.sqrt(v[1]))) for k, v in acc.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for k in range(10):
        src = SRC / f"v{k}.mp4"
        base = OUT / f"m{k}.mp4"
        with tempfile.TemporaryDirectory(dir="E:/envs/tmp") as d:
            paths = extract(src, d)
            w, h = Image.open(paths[0]).size
            for p in paths:  # 제자리 삽입: 읽고 심고 같은 파일에 쓰기
                W.embed(Image.open(p).convert("RGB"), KEY).save(p)
            encode_dir(d, base, 18)
        row = {"v": k}
        for name, crf, scale, cut in [("crf18", 18, None, None), ("crf28", 28, None, None), ("crf35", 35, None, None),
                                      ("480p", 23, (854, 480), None), ("sub2s", 23, None, (3, 2))]:
            q = OUT / f"q{k}_{name}.mp4"
            vf = ["-vf", f"scale={scale[0]}:{scale[1]}"] if scale else []
            cutargs = ["-ss", str(cut[0]), "-t", str(cut[1])] if cut else []
            subprocess.run([FF, "-y", "-loglevel", "error", *cutargs, "-i", str(base), *vf, "-c:v", "libx264", "-crf", str(crf),
                            "-pix_fmt", "yuv420p", str(q)], check=True)
            with tempfile.TemporaryDirectory(dir="E:/envs/tmp") as d:
                qp = extract(q, d, size=(w, h) if scale else None)  # 축소본은 원래 크기로 다시 늘려 블록 격자를 맞춤
                r = zs(qp, [KEY, KEY + 1])  # KEY+1: 음성 대조(틀린 key)
            row[name] = {"z_single_median": r[KEY][0], "z_accum": r[KEY][1], "n": len(qp), "neg_single": r[KEY + 1][0], "neg_accum": r[KEY + 1][1]}
            q.unlink()
        res.append(row)
        with (OUT / "rows.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + chr(10))
        print(k, {n: (round(row[n]["z_single_median"], 1), round(row[n]["z_accum"], 1)) for n in ["crf18", "crf28", "crf35", "480p", "sub2s"]}, flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
