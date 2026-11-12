"""일반 처리 공격: WAVES 단일 왜곡 8종을 PIL·numpy로 옮긴 것.

원본: umd-huang-lab/waves 커밋 1477635, distortions/distortions.py
강도는 WAVES처럼 상대값(0~1)을 받아 각 공격의 범위로 바꾼다.
"""
import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

RANGES = {
    "rotation": (0, 45),
    "resizedcrop": (1, 0.5),
    "erasing": (0, 0.25),
    "brightness": (1, 2),
    "contrast": (1, 2),
    "blurring": (0, 20),
    "noise": (0, 0.1),
    "compression": (90, 10),
}
NAMES_KO = {
    "rotation": "회전", "resizedcrop": "크롭+확대", "erasing": "지우기", "brightness": "밝기",
    "contrast": "대비", "blurring": "흐림", "noise": "잡음", "compression": "JPEG",
}


def absolute(kind, rel):
    a, b = RANGES[kind]
    v = a + rel * (b - a)
    return min(max(v, min(a, b)), max(a, b))


def _square_params(w, h, area_scale, rng):
    # torchvision RandomResizedCrop/RandomErasing.get_params(scale=(s, s), ratio=(1, 1))와 같은 정사각 영역
    side = int(round((area_scale * w * h) ** 0.5))
    side = max(1, min(side, w, h))
    top = rng.randint(0, h - side)
    left = rng.randint(0, w - side)
    return top, left, side


def apply(img, kind, rel, seed=0):
    img = img.convert("RGB")
    s = absolute(kind, rel)
    rng = random.Random(seed)
    w, h = img.size
    if kind == "rotation":
        return img.rotate(s, resample=Image.NEAREST, expand=False, fillcolor=(0, 0, 0))
    if kind == "resizedcrop":
        top, left, side = _square_params(w, h, s, rng)
        return img.crop((left, top, left + side, top + side)).resize((w, h), Image.BILINEAR)
    if kind == "erasing":
        if s <= 0:
            return img.copy()
        top, left, side = _square_params(w, h, s, rng)
        a = np.array(img)
        a[top:top + side, left:left + side] = 0
        return Image.fromarray(a)
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(s)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(s)
    if kind == "blurring":
        return img.filter(ImageFilter.GaussianBlur(int(s)))
    if kind == "noise":
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = a + np.random.default_rng(seed).standard_normal(a.shape).astype(np.float32) * s
        return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))
    if kind == "compression":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(s))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(kind)
