"""순찰기의 음원·영상·문서 판정(monitor.check_media)을 가짜 재게시 페이지 docs/mirror/media.html의 파일 4개에 돌려 결과를 파일로 남긴다.

전체 순찰(그림 + TrustMark)은 메모리가 커서 사용자 확인 대기 — 이 스크립트는 모델을 올리지 않는 매체 판정만 한다.
기대(11-15 10:10 직접 시험): media-1·2·3 찾음, media-4(상관없는 우리 다른 문서 PDF) 못 찾음.
결과: experiments/results/monitor_media.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import monitor  # noqa: E402

FILES = {"media-1.mp3": "MP3 32kbps로 다시 압축한 음원", "media-2.mp4": "좌우 반전해 다시 인코딩한 영상",
         "media-3.docx": "앞부분을 지우고 다시 저장한 docx", "media-4.pdf": "상관없는 연구실 다른 문서(음성 대조군)"}


def main():
    out = {}
    for f, what in FILES.items():
        r = monitor.check_media((ROOT / "docs" / "mirror" / "img" / f).read_bytes(), f)
        out[f] = {"what": what, "found": r is not None, "result": r}
    (ROOT / "experiments" / "results" / "monitor_media.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v["found"] for k, v in out.items()}))


if __name__ == "__main__":
    main()
