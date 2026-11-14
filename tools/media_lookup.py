"""음원·영상 파일을 원장의 fingerprint와 대조한다. C2PA 증명서가 재압축으로 사라진 음원·영상의 원래 기록 찾기.

- 음원: Philips 방식(tools/afp.py), BER < 0.35
- 영상: vPDQ 방식(tools/vfp.py), 질의측 매칭 비율 ≥ 0.8 (잘라 올린 영상도 찾도록 — 실험 21)
사용법: python tools/media_lookup.py <파일(wav, mp3, mp4 등)>
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
import vfp

ROOT = Path(__file__).resolve().parent.parent


def load_audio(path):
    """ffmpeg으로 모노 44.1kHz WAV로 풀어서 읽는다(MP3·AAC 등 어떤 형식이든)."""
    with tempfile.TemporaryDirectory(dir="E:/envs/tmp") as d:
        out = Path(d) / "a.wav"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "44100", str(out)], check=True)
        with wave.open(str(out), "rb") as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32767, w.getframerate()


TEMPO_FACTORS = [0.95, 0.96, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05]  # 실험 25: 속도 ±5%까지 10/10


def _tempo_variants(path):
    """처음 비교에서 못 찾으면 질의 속도를 1% 간격으로 되돌려 본다(음 높이 유지, ffmpeg atempo)."""
    with tempfile.TemporaryDirectory(dir="E:/envs/tmp") as d:
        for f in TEMPO_FACTORS:
            out = Path(d) / f"t{f}.wav"
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-af", f"atempo={f}",
                            "-ac", "1", "-ar", "44100", str(out)], check=True)
            with wave.open(str(out), "rb") as w:
                yield f, afp.fingerprint(np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32767, w.getframerate())


def lookup(path, tempo_search=True):
    r = _lookup_fp(*load_audio(path))
    if r["found"] or not tempo_search:
        return r
    best = r
    for f, fq in _tempo_variants(path):
        rr = _lookup_fp(None, None, fq)
        if rr.get("ber") is not None and (best.get("ber") is None or rr["ber"] < best["ber"]):
            best = dict(rr, tempo_factor=f)
    return best


def _lookup_fp(x, sr, fq=None):
    if fq is None:
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


def lookup_video(path, q_min=0.8):
    hq = vfp.frame_hashes(path)
    best = None
    for e in ledger.load(ROOT / "docs" / "assets" / "ledger.jsonl"):
        r = e["record"]
        if "vpdq" not in r:
            continue
        hr = (np.array([np.unpackbits(np.frombuffer(bytes.fromhex(x), np.uint8)) for x in r["vpdq"]]), np.array(r["vpdq_quality"]))
        pq, pc = vfp.match(hq, hr)
        if best is None or pq > best[0]:
            best = (pq, pc, e)
    if best and best[0] >= q_min:
        return {"found": True, "query_side": best[0], "ref_side": best[1], "index": best[2]["index"],
                "file": best[2]["record"]["file"], "caption": best[2]["record"]["caption"]}
    return {"found": False, "query_side": best[0] if best else None}


if __name__ == "__main__":
    f = sys.argv[1]
    is_video = Path(f).suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".avi")
    print(json.dumps(lookup_video(f) if is_video else lookup(f), ensure_ascii=False, indent=1))
