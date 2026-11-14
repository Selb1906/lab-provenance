"""실험 16 준비: arXiv '건물 에너지' 공개 논문 PDF에서 그림을 추출해 방해 그림 원장을 만든다.

방해 그림은 E:/envs/c2pa-work/exp16/distractors/ 에만 두고 저장소·사이트에 올리지 않는다(재배포 안 함).
설계는 notes/개발일지.md "실험 16 설계".
"""
import hashlib
import io
import re
import time
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

OUT = Path("E:/envs/c2pa-work/exp16")
PDF = OUT / "pdf"
DIS = OUT / "distractors"
UA = "lab-provenance-research/0.1 (academic experiment)"
TARGET = 500
MIN_SIDE = 300


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    PDF.mkdir(parents=True, exist_ok=True)
    DIS.mkdir(parents=True, exist_ok=True)
    seen = {p.stem for p in DIS.glob("*.png")}
    ids = []
    for start in (0, 100):
        xml = get(f"http://export.arxiv.org/api/query?search_query=all:%22building%20energy%22&start={start}&max_results=100").decode("utf-8")
        ids += re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", xml)
        time.sleep(3)  # arXiv API 이용 지침: 요청 사이 3초
    print("papers", len(ids), flush=True)
    for aid in ids:
        if len(seen) >= TARGET:
            break
        f = PDF / (aid.replace("/", "_") + ".pdf")
        try:
            if not f.exists():
                f.write_bytes(get(f"https://arxiv.org/pdf/{aid}"))
                time.sleep(3)
            doc = fitz.open(f)
        except Exception as e:  # noqa: BLE001
            print("skip", aid, e, flush=True)
            continue
        n = 0
        for page in doc:
            for img in page.get_images(full=True):
                try:
                    pix = fitz.Pixmap(doc, img[0])
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                except Exception:  # noqa: BLE001
                    continue
                if max(im.size) < MIN_SIDE:
                    continue
                h = hashlib.sha256(im.tobytes()).hexdigest()[:16]
                if h in seen:
                    continue
                if max(im.size) > 1024:  # 실험 1 집합과 같은 크기 제한(load_set의 긴 변 1024)
                    s = 1024 / max(im.size)
                    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                im.save(DIS / f"{h}.png")
                seen.add(h)
                n += 1
        print(aid, "figures", n, "total", len(seen), flush=True)
    print("done", len(seen), flush=True)


if __name__ == "__main__":
    main()
