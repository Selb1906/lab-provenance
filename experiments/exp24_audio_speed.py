"""실험 24: 음원 지문(Philips 방식)이 속도·음 높이 변화를 얼마나 견디나.

설계·예측은 notes/개발일지.md "실험 24 설계". 결과는 E:/envs/c2pa-work/exp24/summary.json.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import afp  # noqa: E402
from exp20_audio_fp import SR, read_wav, synth, write_wav  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp24")
FF = imageio_ffmpeg.get_ffmpeg_exe()


def transform(y, kind, pct, tmp):
    a, b = tmp / "a.wav", tmp / "b.wav"
    write_wav(a, y)
    r = 1 + pct / 100
    filt = {"tempo": f"atempo={r}",                                  # 속도만(음 높이 유지)
            "pitch": f"asetrate={SR * r:.0f},atempo={1 / r},aresample={SR}",  # 음 높이만(속도 보정)
            "both": f"asetrate={SR * r:.0f},aresample={SR}"}[kind]   # 테이프처럼: 둘 다
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(a), "-af", filt, "-ar", str(SR), "-ac", "1", str(b)], check=True)
    return read_wav(b)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir="E:/envs/tmp"))
    src = [synth(s) for s in range(10)]
    ref = [afp.fingerprint(y, SR) for y in src]
    res = {}
    for kind in ["tempo", "pitch", "both"]:
        for pct in [-5, -2, -1, 1, 2, 5]:
            rows = []
            for i, y in enumerate(src):
                fq = afp.fingerprint(transform(y, kind, pct, tmp), SR)
                bers = [afp.ber(fq, r) for r in ref]
                rows.append({"self": bers[i], "found": bers[i] < afp.BER_T and bers[i] == min(bers)})
            res[f"{kind}/{pct}"] = {"found": sum(r["found"] for r in rows), "n": len(rows),
                                     "self_ber_median": sorted(r["self"] for r in rows)[len(rows) // 2]}
            print(kind, pct, res[f"{kind}/{pct}"], flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
