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
    data = bytearray(Path(src).read_bytes())
    pos = 8
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = bytes(data[pos + 4:pos + 8])
        if ctype == b"IDAT":
            body_start = pos + 8
            # 압축 스트림을 풀어 픽셀 한 바이트를 바꾼 뒤 다시 압축한다 (그래야 그림이 정상으로 열린다)
            raw = bytearray(zlib.decompress(bytes(data[body_start:body_start + length])))
            raw[len(raw) // 2] ^= 0x01
            body = zlib.compress(bytes(raw))
            chunk = struct.pack(">I", len(body)) + b"IDAT" + body
            chunk += struct.pack(">I", zlib.crc32(b"IDAT" + body) & 0xFFFFFFFF)
            data[pos:pos + 12 + length] = chunk
            break
        pos += 12 + length
    Path(dst).write_bytes(bytes(data))


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    resave(src, out / f"{src.stem}-resaved.png")
    tamper_png(src, out / f"{src.stem}-tampered.png")
    print("ok")
