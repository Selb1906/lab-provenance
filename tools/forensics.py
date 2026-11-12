"""수동 포렌식: 파일 자체를 분석해 편집 흔적을 찾는다.

- exif: 메타데이터(촬영 기기, 편집 프로그램, 날짜). 지우기 쉬워 '있으면 참고, 없으면 무의미'.
- ela(Error Level Analysis): 같은 품질로 다시 JPEG 저장했을 때 원본과의 차이. 한 번 압축된 영역과
  나중에 붙인 영역은 다시 압축할 때 오차 크기가 달라 밝기 차이로 드러난다. 원본이 JPEG일 때만 의미가 있다.
- jpeg_quality: 양자화 표로 추정한 저장 품질. 여러 번 저장된 흔적을 볼 때 쓴다.
"""
import io

import numpy as np
from PIL import ExifTags, Image


def exif(img):
    raw = img.getexif()
    return {ExifTags.TAGS.get(k, str(k)): str(v)[:80] for k, v in raw.items()}


def ela(img, quality=90, scale=None):
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    re = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(rgb, np.int16) - np.asarray(re, np.int16)).max(axis=2).astype(np.float64)
    s = scale or (255.0 / max(diff.max(), 1))
    return Image.fromarray(np.clip(diff * s, 0, 255).astype(np.uint8)), diff


def jpeg_quality(img):
    """libjpeg 표준 휘도 양자화 표와 비교해 품질을 추정(IJG 공식의 역산)."""
    q = getattr(img, "quantization", None)
    if not q:
        return None
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
    t = np.array(q[0], float)
    scale = np.median(t * 100 / std)
    return int(round((200 - scale) / 2)) if scale <= 100 else int(round(5000 / scale))
