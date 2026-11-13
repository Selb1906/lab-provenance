"""그림 말고 음원(WAV)·영상(MP4)에도 같은 서명 코드가 그대로 쓰이는지 확인한다 (조사 05 3단계 계획의 1단계).

- 음원: 3초짜리 안내음(합성). 영상: 발행 그림 3장을 2초씩 넘기는 6초 슬라이드(ffmpeg은 imageio-ffmpeg 내장판, E:에 설치).
- 서명은 tools/sign.py 그대로(형식은 파일 확장자로 판단). 결과는 docs/assets/media/.
사용법: python tools/sign_media.py
"""
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from c2pa import Reader

import sign as signer_mod

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "media"
KEYS = Path("E:/envs/c2pa-site-keys")
SR = 44100  # CD 음질 표본율(WAV에서 가장 흔한 값)


def make_wav(path):
    t = np.arange(int(SR * 3)) / SR
    # 세 음(도·미·솔)을 1초씩 — 사람이 들어도 구별되는 내용이면 충분하다
    f = np.where(t < 1, 523.25, np.where(t < 2, 659.25, 783.99))
    y = (0.3 * np.sin(2 * np.pi * f * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


def make_mp4(path):
    figs = sorted((ROOT / "docs" / "assets" / "signed").glob("*.png"))[:3]
    lst = path.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.as_posix()}'\nduration 2\n" for p in figs) + f"file '{figs[-1].as_posix()}'\n", encoding="utf-8")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:white,format=yuv420p",
                    "-r", "30", "-c:v", "libx264", "-movflags", "+faststart", str(path)], check=True)
    lst.unlink()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir="E:/envs/tmp"))
    jobs = [("tones.wav", make_wav, "안내음 3초(도·미·솔)"), ("slides.mp4", make_mp4, "발행 그림 3장 슬라이드 6초")]
    report = []
    for name, make, title in jobs:
        raw = tmp / name
        make(raw)
        dst = OUT / name
        signer_mod.sign(raw, dst, title, "Test media for the C2PA multi-format check. Green Energy Lab, Sangmyung University.", KEYS)
        mime = {"wav": "audio/wav", "mp4": "video/mp4"}[name.split(".")[-1]]
        with open(dst, "rb") as s, Reader(mime, s) as r:
            st = json.loads(r.json())
        report.append({"file": f"assets/media/{name}", "bytes": dst.stat().st_size, "raw_bytes": raw.stat().st_size,
                       "validation_state": st.get("validation_state"),
                       "codes": [v["code"] for v in st.get("validation_status", [])]})
        print(report[-1], file=sys.stderr)
    (OUT / "media.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
