"""자동 발견(모니터링) 시제품: 웹 페이지를 돌며 우리 콘텐츠가 어디서 쓰이는지 찾는다.

시작 주소 목록 → robots.txt 확인 → 페이지의 <img> 수집 → 이미지마다 대조:
  C2PA(증명서 남아 있나) / SHA-256(같은 파일인가) / 지각 해시(같은 그림인가) /
  조각 해시(일부를 잘라 썼나) / TrustMark 식별자(누구에게 간 사본인가)
결과: 어디서 · 어떤 원본이 · 어떻게 바뀌어 쓰였는지.

사용법: python tools/monitor.py <시작 주소> [<시작 주소> ...]
"""
import hashlib
import io
import json
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

import fphash
import ledger

ROOT = Path(__file__).resolve().parent.parent
LEDGERS = [ROOT / "docs" / "assets" / "ledger.jsonl", ROOT / "docs" / "assets" / "journey" / "journey-ledger.jsonl"]
# 받는 사람별 식별자 대응표(능동 포렌식). 실제 운영에서는 비공개 원장에 둔다
RECIPIENTS = {"RCV-A001": "공동연구기관 A", "RCV-B002": "자료 요청자 B", "RCV-C003": "기업 C"}
UA = "lab-provenance-monitor/0.1 (research prototype)"
PHASH_T = 5  # 원장 14장 기준 질의 오탐 1% 이하(실험 9). 원장이 커지면 다시 낮춰야 함
TILE_T = 5  # 조각 비교는 질의당 오탐 1% 기준(실험 8). 조각이 많아 우연히 가까운 것이 생기므로 훨씬 엄격


class _Imgs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.srcs = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            src = dict(attrs).get("src")
            if src:
                self.srcs.append(src)


# 증거 보존(조사 12): 발견 시점의 응답 헤더 중 날짜·서버·형식만 남긴다(쿠키 등 개인 정보가 섞일 수 있는 헤더는 버림)
KEEP_HEADERS = ("Date", "Last-Modified", "Content-Type", "Content-Length", "Server", "ETag")


def _get(url, meta=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
        if meta is not None:
            meta.update({"url": url, "status": r.status, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "headers": {k: r.headers[k] for k in KEEP_HEADERS if r.headers.get(k)}})
        return data


def _allowed(url, cache={}):
    p = urllib.parse.urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    if base not in cache:
        rp = urllib.robotparser.RobotFileParser(base + "/robots.txt")
        try:
            rp.read()  # 404 등 파일 없음은 '전부 허용', 401·403은 '전부 금지'로 표준 라이브러리가 처리
        except Exception:
            # 연결 실패 등으로 robots.txt를 확인할 수 없으면 '전부 금지'로 본다 (RFC 9309 2.3.1.4)
            rp = False
        cache[base] = rp
    rp = cache[base]
    return False if rp is False else rp.can_fetch(UA, url)


def _load_refs():
    mains, tiles, wm = [], [], {}
    for lp in LEDGERS:
        for e in ledger.load(lp):
            r = e["record"]
            if r.get("type") == "tiles":
                tiles.append(r)
            elif "lab_phash" in r:
                mains.append(r)
                if r.get("watermark_id"):
                    wm[r["watermark_id"]] = r
    return mains, tiles, wm


def _c2pa(data):
    from c2pa import Context, Reader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as t:
        t.write(data)
        name = t.name
    try:
        fmt = (Image.open(io.BytesIO(data)).format or "png").lower()
        with Context() as ctx, open(name, "rb") as s, Reader(f"image/{fmt}", s, context=ctx) as r:
            st = json.loads(r.json())
        codes = [v["code"] for v in st.get("validation_status", [])]
        return "변조" if any("mismatch" in c for c in codes) else "남아 있음"
    except Exception:
        return "없음"
    finally:
        Path(name).unlink(missing_ok=True)


def _tm():
    from trustmark import TrustMark
    return TrustMark(verbose=False, model_type="Q", loadRemover=False, loadBBoxDetector=True)


def check_image(data, refs, tm):
    mains, tiles, wm = refs
    im = Image.open(io.BytesIO(data)).convert("RGB")
    sha = hashlib.sha256(data).hexdigest()
    ph = fphash.to_hex(fphash.lab_phash(im))
    exact = next((m for m in mains if m.get("sha256") == sha), None)
    near = min(((fphash.distance(ph, m["lab_phash"]), m) for m in mains), key=lambda x: x[0])
    tile = min(((fphash.distance(ph, t), r) for r in tiles for t in r["tiles"]), key=lambda x: x[0]) if tiles else (99, None)
    s1, ok1, _ = tm.decode(im, MODE="text")
    s2, ok2, _ = tm.decode(im, MODE="text", DETECTFIRST=True)
    wid = s1 if ok1 else (s2 if ok2 else None)
    how, src = [], None
    if exact:
        src, how = exact, ["원본 파일 그대로"]
    elif near[0] <= PHASH_T:
        src, how = near[1], [f"다시 저장·압축됨(지각 해시 거리 {near[0]})"]
    elif tile[0] <= TILE_T:
        src = next((m for m in mains if m.get("sha256") == tile[1]["of_sha256"]), None)
        how = [f"일부를 잘라 씀(조각 해시 거리 {tile[0]})"]
    if wid in wm:
        src = src or wm[wid]
        how.append(f"워터마크 {wid} 확인")
    elif wid in RECIPIENTS:
        how.append(f"워터마크 {wid}: {RECIPIENTS[wid]}에게 보낸 사본")
        src = src or {"file": "(받는 사람별 사본)", "caption": "받는 사람별 식별자를 심어 보낸 사본"}
    if not src and not how:
        return None
    return {"c2pa": _c2pa(data), "source": (src or {}).get("file"), "caption": (src or {}).get("caption"),
            "how": how, "phash_d": near[0], "tile_d": tile[0], "watermark": wid}


def crawl(starts):
    refs = _load_refs()
    tm = _tm()
    found, checked = [], 0
    for page in starts:
        if not _allowed(page):
            print("robots.txt disallows", page)
            continue
        p = _Imgs()
        page_meta = {}
        p.feed(_get(page, page_meta).decode("utf-8", "replace"))
        for src in p.srcs:
            url = urllib.parse.urljoin(page, src)
            if not _allowed(url):
                continue
            checked += 1
            img_meta = {}
            hit = check_image(_get(url, img_meta), refs, tm)
            if hit:
                found.append({"page": page, "image": url, **hit, "evidence": {"page": page_meta, "image": img_meta}})
            time.sleep(0.5)  # 상대 서버에 부담을 주지 않도록 요청 간격
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pages": starts, "images_checked": checked, "found": found}


if __name__ == "__main__":
    rep = crawl(sys.argv[1:])
    out = ROOT / "docs" / "assets" / "monitor-report.json"
    text = json.dumps(rep, ensure_ascii=False, indent=1)
    out.write_text(text, encoding="utf-8", newline=chr(10))
    # 보고서 자체의 해시를 원장에 남긴다 → 다음 비트코인 고정(anchor.py stamp)에 함께 묶여 '이 시각에 이렇게 발견했다'가 고정된다
    if rep["found"]:
        e = ledger.append(LEDGERS[0], {"type": "monitor-evidence", "report_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                       "time": rep["time"], "n_found": len(rep["found"])})
        print("ledger #", e["index"], file=sys.stderr)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
