"""검증 페이지 시연용 변형 예시를 만든다.

- resaved: 이미지 라이브러리로 다시 저장 (일반 편집기·SNS처럼 모르는 메타데이터를 버린다)
- tampered: 가운데 픽셀 하나의 값을 1 바꾸어 IDAT만 새로 쓰고 CRC를 맞춘다 (증명서는 그대로 남는다)
"""
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


def resave(src, dst):
    Image.open(src).save(dst)


def tamper_png(src, dst):
    data = bytes(Path(src).read_bytes())
    chunks, pos = [], 8
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunks.append((data[pos + 4:pos + 8], data[pos:pos + 12 + length], data[pos + 8:pos + 8 + length]))
        pos += 12 + length
    # 픽셀 데이터는 IDAT 여러 개로 나뉠 수 있다 → 모두 이어 붙여 풀고, 한 바이트를 바꾼 뒤 IDAT 하나로 다시 넣는다
    # 압축 해제한 바이트를 그대로 바꾸면 PNG 행 필터(Sub·Up·Paeth) 때문에 그 뒤 픽셀이 줄줄이 바뀐다
    # (처음 판은 이 때문에 지각 해시 거리가 16까지 벌어져 '1바이트 변조'가 아니었음, 11-13 발견).
    # 그래서 픽셀을 풀어 가운데 한 픽셀의 한 채널만 1 바꾸고, 모든 행을 필터 0(그대로)으로 다시 묶는다.
    ihdr = next(body for t, _, body in chunks if t == b"IHDR")
    assert ihdr[8] == 8, "8비트 PNG만 지원"
    im = Image.open(src)
    arr = np.array(im)
    y, x = arr.shape[0] // 2, arr.shape[1] // 2
    if arr.ndim == 2:
        arr[y, x] ^= 1
    else:
        arr[y, x, 0] ^= 1
    rows = arr.reshape(arr.shape[0], -1)
    raw = b"".join(bytes(1) + r.tobytes() for r in rows)  # 행마다 필터 종류 0(None)
    body = zlib.compress(raw)
    idat = struct.pack(">I", len(body)) + b"IDAT" + body + struct.pack(">I", zlib.crc32(b"IDAT" + body) & 0xFFFFFFFF)
    out, done = data[:8], False
    for t, whole, _ in chunks:
        if t == b"IDAT":
            if not done:
                out += idat
                done = True
            continue
        out += whole
    Path(dst).write_bytes(out)


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    resave(src, out / f"{src.stem}-resaved.png")
    tamper_png(src, out / f"{src.stem}-tampered.png")
    print("ok")
