"""실험 25: 질의 음원의 속도를 거꾸로 여러 단계로 되돌려 보고 BER 최소를 쓰면 속도 ±5%도 찾는가(다중 비교 오탐 확인 포함).

설계·예측은 notes/개발일지.md "실험 25 설계". 결과는 E:/envs/c2pa-work/exp25/summary.json.
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
from exp24_audio_speed import transform  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp25")
FF = imageio_ffmpeg.get_ffmpeg_exe()
FACTORS = [0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05]


def search_fps(y, tmp):
    """질의를 각 속도 후보로 되돌린 fingerprint 목록."""
    out = []
    for f in FACTORS:
        if f == 1.0:
            out.append(afp.fingerprint(y, SR))
            continue
        a, b = tmp / "s.wav", tmp / "t.wav"
        write_wav(a, y)
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(a), "-af", f"atempo={f}", "-ar", str(SR), "-ac", "1", str(b)], check=True)
        out.append(afp.fingerprint(read_wav(b)[0], SR))
    return out


def best_ber(qfps, ref):
    return min(afp.ber(q, ref) for q in qfps)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir="E:/envs/tmp"))
    src = [synth(s) for s in range(10)]
    ref = [afp.fingerprint(y, SR) for y in src]
    res = {}
    for kind in ["tempo", "both"]:
        for pct in [-5, -2, 2, 5]:
            found = 0
            sel = []
            for i, y in enumerate(src):
                qf = search_fps(transform(y, kind, pct, tmp), tmp)
                bers = [best_ber(qf, r) for r in ref]
                ok = bers[i] < afp.BER_T and bers[i] == min(bers)
                found += ok
                sel.append(bers[i])
            res[f"{kind}/{pct}"] = {"found": found, "n": 10, "self_ber_median": sorted(sel)[5]}
            print(kind, pct, res[f"{kind}/{pct}"], flush=True)
    imp = []
    for i in range(10):
        qf = search_fps(src[i], tmp)
        imp += [best_ber(qf, ref[j]) for j in range(10) if j != i]
    res["impostor_min_ber_with_search"] = min(imp)
    res["impostor_min_ber_without_search"] = min(afp.ber(ref[i], ref[j]) for i in range(10) for j in range(10) if i != j)
    print("impostor min with/without search", res["impostor_min_ber_with_search"], res["impostor_min_ber_without_search"], flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
