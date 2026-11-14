"""음원 파일을 원장의 음원 fingerprint(Philips 방식, tools/afp.py)와 대조한다. C2PA 증명서가 재압축으로 사라진 음원의 원래 기록 찾기.

사용법: python tools/audio_lookup.py <음원 파일(wav, mp3 등)>
"""
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

import afp
import ledger

ROOT = Path(__file__).resolve().parent.parent


def load_audio(path):
    """ffmpeg으로 모노 44.1kHz WAV로 풀어서 읽는다(MP3·AAC 등 어떤 형식이든)."""
    with tempfile.TemporaryDirectory(dir="E:/envs/tmp") as d:
        out = Path(d) / "a.wav"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "44100", str(out)], check=True)
        with wave.open(str(out), "rb") as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32767, w.getframerate()


def lookup(path):
    x, sr = load_audio(path)
    fq = afp.fingerprint(x, sr)
    best = None
    for e in ledger.load(ROOT / "docs" / "assets" / "ledger.jsonl"):
        r = e["record"]
        if "afp_philips" not in r:
            continue
        fr = np.unpackbits(np.frombuffer(bytes.fromhex(r["afp_philips"]), np.uint8).reshape(r["afp_frames"], 4), axis=1)
        b = afp.ber(fq, fr)
        if best is None or b < best[0]:
            best = (b, e)
    if best and best[0] < afp.BER_T:
        return {"found": True, "ber": best[0], "index": best[1]["index"], "file": best[1]["record"]["file"], "caption": best[1]["record"]["caption"]}
    return {"found": False, "ber": best[0] if best else None}


if __name__ == "__main__":
    print(json.dumps(lookup(sys.argv[1]), ensure_ascii=False, indent=1))
