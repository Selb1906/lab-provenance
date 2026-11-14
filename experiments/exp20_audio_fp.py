"""실험 20: Philips 음원 fingerprint로 MP3 재압축·음량·잡음 뒤에도 원본 음원을 찾는가.

설계·예측은 notes/개발일지.md "실험 20 설계". 결과는 E:/envs/c2pa-work/exp20/summary.json.
"""
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import afp  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp20")
SR = 44100
FF = imageio_ffmpeg.get_ffmpeg_exe()


def synth(seed, sec=10):
    """서로 다른 합성 음원: seed마다 음계·리듬·화음·잡음 색이 다르다(공개 음원을 내려받지 않기 위해)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * sec)) / SR
    y = np.zeros_like(t)
    beat = rng.uniform(0.15, 0.5)
    notes = 220 * 2 ** (rng.integers(0, 24, size=int(sec / beat) + 1) / 12)
    for k, f in enumerate(notes):
        s, e = int(k * beat * SR), min(int((k + 1) * beat * SR), len(t))
        if s >= len(t):
            break
        seg = t[s:e] - t[s]
        env = np.exp(-seg * rng.uniform(2, 8))
        y[s:e] += env * (np.sin(2 * np.pi * f * seg) + 0.4 * np.sin(2 * np.pi * 2 * f * seg) + 0.2 * np.sin(2 * np.pi * 1.5 * f * seg))
    y += 0.02 * np.cumsum(rng.standard_normal(len(t))) / np.sqrt(len(t))  # 저주파 잡음(음원마다 다름)
    return 0.3 * y / np.max(np.abs(y))


def write_wav(path, y):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32767
        if w.getnchannels() > 1:
            d = d.reshape(-1, w.getnchannels()).mean(axis=1)
    return d, sr


def mp3_roundtrip(y, kbps, tmp):
    a, b, c = tmp / "a.wav", tmp / "b.mp3", tmp / "c.wav"
    write_wav(a, y)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(a), "-b:a", f"{kbps}k", str(b)], check=True)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(b), "-ar", str(SR), "-ac", "1", str(c)], check=True)
    return read_wav(c)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir="E:/envs/tmp"))
    N = 10
    src = [synth(s) for s in range(N)]
    ref = [afp.fingerprint(y, SR) for y in src]
    res = {"genuine": {}, "impostor": []}
    for i, y in enumerate(src):
        rng = np.random.default_rng(100 + i)
        atk = {"mp3_128": mp3_roundtrip(y, 128, tmp), "mp3_64": mp3_roundtrip(y, 64, tmp), "mp3_32": mp3_roundtrip(y, 32, tmp),
               "gain_-6dB": y * 10 ** (-6 / 20),
               "noise_20dB": y + rng.standard_normal(len(y)) * np.sqrt(np.mean(y ** 2) / 10 ** (20 / 10))}
        for k, q in atk.items():
            fq = afp.fingerprint(q, SR)
            bers = [afp.ber(fq, r) for r in ref]
            res["genuine"].setdefault(k, []).append({"self": bers[i], "best_other": min(b for j, b in enumerate(bers) if j != i)})
        print(i, {k: round(v[-1]["self"], 3) for k, v in res["genuine"].items()}, flush=True)
    for i in range(N):
        for j in range(i + 1, N):
            res["impostor"].append(afp.ber(ref[i], ref[j]))
    summ = {k: {"self_ber_max": max(x["self"] for x in v), "found": sum(x["self"] < afp.BER_T and x["self"] < x["best_other"] for x in v),
                "n": len(v)} for k, v in res["genuine"].items()}
    summ["impostor_ber_min"] = min(res["impostor"])
    res["summary"] = summ
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
