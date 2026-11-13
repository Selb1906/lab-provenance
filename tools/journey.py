"""'그림 한 장의 여정' 페이지용 자료를 만든다.

연구실이 그림 하나를 발행할 때 모든 층을 다 쓴다: TrustMark 워터마크(식별자) → C2PA 서명 → 원장 등록.
그 뒤 그림이 겪는 일을 실제로 만들어 보고, 각 층이 무엇을 말해 주는지 검사한 결과를 JSON으로 남긴다.
공격 강도는 WAVES 값을 쓴다(개발일지 실험 1 설계). 화면 캡처 이미지는 Playwright로 따로 만들어 넣는다.

사용법: python tools/journey.py prepare   # 발행본·변형본 생성
        python tools/journey.py check     # 화면 캡처까지 넣은 뒤 전 층 검사
"""
import hashlib
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

import attacks
import fphash
import ledger
import sign as signer_mod

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "journey"
KEYS = Path("E:/envs/c2pa-site-keys")
SRC = ROOT / "docs" / "assets" / "signed" / "buildings2026-02.png"  # 첫 화면의 히트맵 그림
JLEDGER = OUT / "journey-ledger.jsonl"
WM_ID = "GEL-0002"
RECIPIENTS = {"RCV-A001": "공동연구기관 A", "RCV-B002": "학회 발표 자료 요청자 B", "RCV-C003": "기업 C"}


def _tm():
    from trustmark import TrustMark
    return TrustMark(verbose=False, model_type="Q", loadRemover=False, loadBBoxDetector=True)


def _png_pixels_replace(signed_path, new_rgb, dst):
    """서명된 PNG의 픽셀(IDAT)만 바꾸고 증명서(caBX)는 그대로 둔다 — 증명서를 남긴 채 내용을 조작하는 경우."""
    data = bytearray(Path(signed_path).read_bytes())
    pos, chunks = 8, []
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        chunks.append((bytes(data[pos + 4:pos + 8]), pos, n))
        pos += 12 + n
    ihdr = next(c for c in chunks if c[0] == b"IHDR")
    w, h = struct.unpack(">II", data[ihdr[1] + 8:ihdr[1] + 16])
    arr = np.asarray(new_rgb.convert("RGB"), np.uint8)
    assert arr.shape[:2] == (h, w)
    raw = b"".join(b"\x00" + arr[r].tobytes() for r in range(h))  # 필터 0, 8비트 RGB (서명본과 같은 형식일 때만 유효)
    body = zlib.compress(raw, 9)
    idat = struct.pack(">I", len(body)) + b"IDAT" + body + struct.pack(">I", zlib.crc32(b"IDAT" + body) & 0xFFFFFFFF)
    first = next(c for c in chunks if c[0] == b"IDAT")
    out = bytes(data[:first[1]]) + idat
    rest = [c for c in chunks if c[1] > first[1] and c[0] != b"IDAT"]
    for c in rest:
        out += bytes(data[c[1]:c[1] + 12 + c[2]])
    Path(dst).write_bytes(out)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    tm = _tm()
    base = Image.open(SRC).convert("RGB")
    # 1) 발행: 워터마크 → 서명 → 원장
    wm = tm.encode(base, WM_ID, MODE="text")
    wm.save(OUT / "_wm.png")
    signer_mod.sign(OUT / "_wm.png", OUT / "0-published.png", "건물 유형별 시간대 오차 패턴 (여정 시연본)",
                    f"Watermark id {WM_ID}. Published by Green Energy Lab, Sangmyung University.", KEYS)
    (OUT / "_wm.png").unlink()
    pub = OUT / "0-published.png"
    if JLEDGER.exists():
        JLEDGER.unlink()
    from c2pa import Context, Reader
    with Context() as ctx, Reader(str(pub), context=ctx) as r:
        mid = json.loads(r.json())["active_manifest"]
    ledger.append(JLEDGER, {"file": "assets/journey/0-published.png", "sha256": hashlib.sha256(pub.read_bytes()).hexdigest(),
                            "lab_phash": fphash.to_hex(fphash.lab_phash(Image.open(pub))), "watermark_id": WM_ID,
                            "c2pa_manifest": mid})
    p = Image.open(pub).convert("RGB")
    # 2) 메신저로 전달: JPEG 재압축 (WAVES compression 0.2 = 품질 74)
    attacks.apply(p, "compression", 0.2).save(OUT / "1-messenger.jpg", quality=95)
    # 3) 잘라서 사용: WAVES resizedcrop 0.4 (면적 0.8)
    attacks.apply(p, "resizedcrop", 0.4, seed=7).save(OUT / "2-cropped.png")
    # 4) 화면 캡처: Playwright로 0-published.png 를 탭에서 열어 캡처 → 3-screenshot.png (별도 단계)
    # 5) 내용 조작: 히트맵 한 칸을 다른 색으로 칠하고 증명서는 그대로 둔다
    a = np.asarray(p).copy()
    h, w = a.shape[:2]
    y0, x0 = int(h * 0.62), int(w * 0.08)
    a[y0:y0 + int(h * 0.05), x0:x0 + int(w * 0.06)] = (178, 24, 43)
    _png_pixels_replace(pub, Image.fromarray(a), OUT / "4-tampered.png")
    # 6) 유출: 받는 사람마다 다른 식별자를 심은 사본 중 B의 것이 메신저 재압축을 거쳐 퍼짐
    for rid in RECIPIENTS:
        tm.encode(base, rid, MODE="text").save(OUT / f"_copy_{rid}.png")
    attacks.apply(Image.open(OUT / "_copy_RCV-B002.png"), "compression", 0.2).save(OUT / "5-leaked.jpg", quality=95)
    for rid in RECIPIENTS:
        (OUT / f"_copy_{rid}.png").unlink()
    print("prepared", sorted(x.name for x in OUT.iterdir()))


def _c2pa(path):
    from c2pa import Context, Reader
    try:
        with Context() as ctx, Reader(str(path), context=ctx) as r:
            s = json.loads(r.json())
        codes = [v["code"] for v in s.get("validation_status", [])]
        if any("mismatch" in c for c in codes):
            return "변조됨", codes
        return "유효(신뢰 목록 밖)", codes
    except Exception:
        return "증명서 없음", []


def check():
    tm = _tm()
    L = ledger.load(JLEDGER)[0]["record"]
    from exp03_geo_fingerprint import orb, orb_score  # noqa: E402
    pub = Image.open(OUT / "0-published.png").convert("RGB")
    O = orb(pub)
    res = {}
    for f in ["0-published.png", "1-messenger.jpg", "2-cropped.png", "3-screenshot.png", "4-tampered.png", "5-leaked.jpg"]:
        p = OUT / f
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        c2, codes = _c2pa(p)
        sha_hit = hashlib.sha256(p.read_bytes()).hexdigest() == L["sha256"]
        d = fphash.distance(fphash.to_hex(fphash.lab_phash(im)), L["lab_phash"])
        s1, ok1, _ = tm.decode(im, MODE="text")
        s2, ok2, _ = tm.decode(im, MODE="text", DETECTFIRST=True)
        wid = s1 if ok1 else (s2 if ok2 else None)
        inl = orb_score(orb(im), O)
        res[f] = {"c2pa": c2, "codes": codes, "sha256_match": sha_hit, "phash_d": d, "phash_ok": d <= 15,
                  "watermark": wid, "orb_inliers": inl, "orb_ok": inl >= 38}
        print(f, res[f])
    res["ledger_ok"] = ledger.verify(JLEDGER) is None
    # 원장 위조 시도: 사본을 만들어 한 줄의 기록을 고친 뒤 검증
    fake = OUT / "_fake.jsonl"
    lines = JLEDGER.read_text(encoding="utf-8").splitlines()
    e = json.loads(lines[0])
    e["record"]["watermark_id"] = "XXX-9999"
    fake.write_text(json.dumps(e, ensure_ascii=False) + "\n", encoding="utf-8")
    res["ledger_tamper_detected_at"] = ledger.verify(fake)
    fake.unlink()
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    {"prepare": prepare, "check": check}[sys.argv[1]]()
