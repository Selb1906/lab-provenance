"""검증 페이지 시연용 변형 예시를 만든다.

- resaved: 이미지 라이브러리로 다시 저장 (일반 편집기·SNS처럼 모르는 메타데이터를 버린다)
- tampered: PNG 픽셀 데이터(IDAT) 한 바이트만 바꾸고 CRC를 다시 맞춘다 (증명서는 그대로 남는다)
"""
import struct
import sys
import zlib
from pathlib import Path

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
    raw = bytearray(zlib.decompress(b"".join(body for t, _, body in chunks if t == b"IDAT")))
    raw[len(raw) // 2] ^= 0x01
    body = zlib.compress(bytes(raw))
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
