"""순찰 실습용 가짜 재게시 이미지 중 '발행 그림을 변형해 올린 것' 4장을 현재 발행본에서 다시 만든다.

처음 판(11-13)은 손으로 만들어 재현 스크립트가 없었고, 워터마크 키를 바꾼 뒤(11-13 22:05) 옛 발행본에서 나온 사본이 되어
새 키로는 워터마크를 읽을 수 없었다. 변형 종류·세기는 처음 판(truth.json)과 같고, seed는 0으로 고정한다.
상관없는 이미지 3장과 여정(journey) 사본 2장은 그대로 둔다.
사용법: python tools/make_mirror.py
"""
from pathlib import Path

from PIL import Image

import attacks

ROOT = Path(__file__).resolve().parent.parent
SIGNED = ROOT / "docs" / "assets" / "signed"
IMG = ROOT / "docs" / "mirror" / "img"

# (파일, 원본, 크롭 WAVES 상대 세기 또는 None, JPEG 품질 또는 None) — truth.json의 'how'와 같은 값.
# JPEG 품질 74·58은 WAVES 압축 0.2·0.4(품질 90→10 구간)와 같다. JPEG은 한 번만 그 품질로 저장한다.
JOBS = [
    ("blog-1.jpg", "kiaebs2026-05", None, 74),
    ("blog-2.png", "buildings2026-01", 0.4, None),
    ("forum-2.jpg", "energy2024-06", None, 58),
    ("news-1.jpg", "enbuild2024-08", 0.8, 74),
]


def main():
    for name, src, crop, q in JOBS:
        im = Image.open(SIGNED / f"{src}.png").convert("RGB")
        if crop is not None:
            im = attacks.apply(im, "resizedcrop", crop, seed=0)
        if q is not None:
            im.save(IMG / name, "JPEG", quality=q)
        else:
            im.save(IMG / name)
        print(name, im.size)


if __name__ == "__main__":
    main()
