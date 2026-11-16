"""자동 발견(모니터링) 시제품: 웹 페이지를 돌며 우리 콘텐츠가 어디서 쓰이는지 찾는다.

시작 주소 목록 → robots.txt 확인 → 페이지의 <img> 수집 → 이미지마다 대조:
  C2PA(증명서 남아 있나) / SHA-256(같은 파일인가) / perceptual hash(같은 그림인가) /
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
import wm_blockdct as W

ROOT = Path(__file__).resolve().parent.parent
LEDGERS = [ROOT / "docs" / "assets" / "ledger.jsonl", ROOT / "docs" / "assets" / "journey" / "journey-ledger.jsonl"]
# 받는 사람별 식별자 대응표(능동 forensics). 실제 운영에서는 비공개 ledger에 둔다
RECIPIENTS = {"RCV-A001": "공동연구기관 A", "RCV-B002": "자료 요청자 B", "RCV-C003": "기업 C"}
UA = "lab-provenance-monitor/0.1 (research prototype)"
PHASH_T = 2  # ledger 해시 14개 × 그림-그림 쌍 FPR 1/1,891 ≤ query FPR 1% (실험 9, 조사 14). 거리는 짝수라 2=3
TILE_T = 5
SIGNED = ROOT / "docs" / "assets" / "signed"
WM_KEY_FILE = Path("E:/envs/c2pa-site-keys/wm_key.txt")  # 블록 DCT 키(비공개). 없으면 정합 단계는 TrustMark만 본다  # 조각 비교는 질의당 false positive 1% 기준(실험 8). 조각이 많아 우연히 가까운 것이 생기므로 훨씬 엄격


class _Imgs(HTMLParser):
    """그림(<img>)과 음원·영상(<audio>·<video>·<source>), 문서 링크(.pdf·.docx)를 모은다."""
    def __init__(self):
        super().__init__()
        self.srcs, self.media, self.docs = [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img" and a.get("src"):
            self.srcs.append(a["src"])
        elif tag in ("audio", "video", "source") and a.get("src"):
            self.media.append(a["src"])
        elif tag == "a" and a.get("href", "").lower().split("?")[0].endswith((".pdf", ".docx")):
            self.docs.append(a["href"])


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
    # 정합 단계용: 지금 발행본 원본과 그 ORB 특징(발행자 쪽에서 돌리므로 원본을 갖고 있다)
    import align
    figs = json.loads((ROOT / "docs" / "assets" / "figures.json").read_text(encoding="utf-8"))
    origs = {}
    for f in figs:
        im = Image.open(ROOT / "docs" / f["file"]).convert("RGB")
        origs[f["file"]] = {"im": im, "orb": align.orb_features(im), "fig": f}
    return mains, tiles, wm, origs


SSCD_ABOVE = int(__import__("os").environ.get("MONITOR_SSCD_ABOVE", "20"))  # 이보다 ledger가 크면 SSCD 후보 단계 사용
SSCD_K = 5
PHASH_ALONE_MAX = 3  # perceptual hash 단독 판정이 query false positive 1%를 지키는 ledger 해시 수 상한 — 95% 상한 기준(1,891쌍 중 1쌍의 Clopper-Pearson 상한 0.29% → 약 3, 조사 15). 점추정이면 18. 지금 ledger(14)은 넘으므로 perceptual hash 일치도 정합 후 watermark로 확인
_SSCD = {}


def _sscd_top(im, origs, k):
    """SSCD(sscd_disc_mixup) 조각 embedding 코사인 상위 k개 원본 이름. 모델·ledger embedding은 처음 한 번만 계산."""
    import numpy as np
    import torch
    from torchvision import transforms
    if "model" not in _SSCD:
        _SSCD["model"] = torch.jit.load("E:/envs/c2pa-work/models/sscd/sscd_disc_mixup.torchscript.pt", map_location="cpu").eval()
        _SSCD["prep"] = transforms.Compose([transforms.Resize(288), transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def tiles(x):
        w, h = x.size
        return [x.crop(b) for b in [(0, 0, w, h), (0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h),
                                    (w // 2, h // 2, w, h), (w // 4, h // 4, 3 * w // 4, 3 * h // 4)]]

    @torch.no_grad()
    def emb(ims):
        return np.stack([torch.nn.functional.normalize(_SSCD["model"](_SSCD["prep"](x.convert("RGB")).unsqueeze(0)), dim=1).numpy()[0] for x in ims])

    names = list(origs)
    if _SSCD.get("names") != names:
        _SSCD["ref"], _SSCD["names"] = np.stack([emb(tiles(origs[n]["im"])) for n in names]), names
    s = np.einsum("qd,ntd->nqt", emb(tiles(im)), _SSCD["ref"]).max(axis=(1, 2))
    return [names[j] for j in np.argsort(-s)[:k]]


def _aligned_watermark(im, origs, tm):
    """앞 단계에서 못 찾았을 때: 후보 원본을 고르고(ORB, 눌러 보기 포함) 정합해 되돌린 뒤 watermark를 읽는다(실험 11b·12).
    판정은 watermark(블록 DCT z > 2.326 또는 TrustMark 식별자가 후보의 식별자와 일치)로만 한다."""
    import align
    pool = list(origs.items())
    if len(pool) > SSCD_ABOVE:
        # ledger가 크면 ORB를 전부와 맞추지 않고 SSCD 상위 K개 후보만 확인(실험 14: 상위 5개 적중 760/760)
        keep = set(_sscd_top(im, origs, SSCD_K))
        pool = [(k, v) for k, v in pool if k in keep]
    name, n_orb = align.candidate(im, [(k, v["orb"]) for k, v in pool])
    if name is None:
        return None
    o = origs[name]
    r = align.register(im, o["im"])
    if r is None:
        return None
    back, valid, _, n_sift = r
    z = (align.bdct_z_valid(back, valid, W.derive_key(int(WM_KEY_FILE.read_text().strip()), o["fig"]["watermark_id"]))
         if WM_KEY_FILE.exists() else None)  # 후보 원본의 식별자로 그림별 key 유도(실험 18)
    s, ok, _ = tm.decode(back, MODE="text")
    tm_ok = bool(ok and s == o["fig"]["watermark_id"])
    if not ((z is not None and z > align.Z_T) or tm_ok):
        return None
    parts = [f"잘라 쓰거나 돌려 씀 — 원본에 맞춰 되돌린 뒤 watermark 확인(특징점 {n_orb}/{n_sift}개 일치"]
    if z is not None:
        parts.append(f"블록 DCT z {z:.1f}")
    if tm_ok:
        parts.append(f"TrustMark {s}")
    return o["fig"], ", ".join(parts) + ")", z, (s if tm_ok else None)


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


class _LazyTM:
    """TrustMark를 처음 필요할 때만 올린다. 위치 검출 모델(BBoxDetector)은 일반 판독이 실패했을 때만 따로 올린다.
    처음 판은 순찰 시작 때 둘을 함께 올려(수백 MB) 음원·영상·문서만 있는 페이지에서도 메모리를 썼고,
    11-14~15에 시스템 메모리 부족 자동 중지가 세 번 났다."""
    def __init__(self):
        self._plain = self._bbox = None

    def decode(self, im, MODE="text", DETECTFIRST=False):
        from trustmark import TrustMark
        if DETECTFIRST:
            if self._bbox is None:
                self._bbox = TrustMark(verbose=False, model_type="Q", loadRemover=False, loadBBoxDetector=True)
            return self._bbox.decode(im, MODE=MODE, DETECTFIRST=True)
        if self._plain is None:
            self._plain = TrustMark(verbose=False, model_type="Q", loadRemover=False)
        return self._plain.decode(im, MODE=MODE)


def _tm():
    return _LazyTM()


def check_image(data, refs, tm):
    mains, tiles, wm, origs = refs
    im = Image.open(io.BytesIO(data)).convert("RGB")
    sha = hashlib.sha256(data).hexdigest()
    ph = fphash.to_hex(fphash.lab_phash(im))
    exact = next((m for m in mains if m.get("sha256") == sha), None)
    near = min(((fphash.distance(ph, m["lab_phash"]), m) for m in mains), key=lambda x: x[0])
    tile = min(((fphash.distance(ph, t), r) for r in tiles for t in r["tiles"]), key=lambda x: x[0]) if tiles else (99, None)
    s1, ok1, _ = tm.decode(im, MODE="text")
    s2, ok2 = (None, False) if ok1 else tm.decode(im, MODE="text", DETECTFIRST=True)[:2]  # 위치 검출은 일반 판독 실패 때만
    wid = s1 if ok1 else (s2 if ok2 else None)
    how, src, decided = [], None, None  # decided: 실제로 출처를 정한 단계(근거 등급에 그 단계만 표시 — 조사 15 지적 4)
    phash_cand = None  # ledger가 커서 perceptual hash만으로 확정 못 할 때의 후보(watermark로 확인되면 확정)
    if exact:
        src, how = exact, ["원본 파일 그대로"]
    elif near[0] <= PHASH_T and len({m["lab_phash"] for m in mains}) <= PHASH_ALONE_MAX:
        src, how, decided = near[1], [f"다시 저장·압축됨(perceptual hash 거리 {near[0]})"], "near"
    elif near[0] <= PHASH_T:
        # ledger가 커서 perceptual hash 단독 판정의 false positive를 보증할 수 없음 → 후보로만 두고 watermark로 확인될 때 확정(아래)
        phash_cand = near[1]
    elif tile[0] <= TILE_T and not WM_KEY_FILE.exists():
        # 조각 해시는 실제 crop에서 0~45%만 찾고(실험 19) false positive만 더한다. 정합 단계(특징점 82~100%)를 쓸 수 있는
        # 발행자 쪽 monitor에서는 조각 해시 일치를 확정 근거로 쓰지 않고 정합 후 watermark로 확인한다(아래 단계).
        src = next((m for m in mains if m.get("sha256") == tile[1]["of_sha256"]), None)
        how, decided = [f"일부를 잘라 씀(조각 해시 거리 {tile[0]})"], "tile"
    if wid in wm:
        src = src or wm[wid]
        how.append(f"watermark {wid} 확인")
    elif wid in RECIPIENTS:
        how.append(f"watermark {wid}: {RECIPIENTS[wid]}에게 보낸 사본")
        src = src or {"file": "(받는 사람별 사본)", "caption": "받는 사람별 식별자를 심어 보낸 사본"}
    aligned_z = None
    if phash_cand is not None and decided is None:
        msg = f"다시 저장·압축됨(perceptual hash 거리 {near[0]})"
        if wid and wid == phash_cand.get("watermark_id"):
            src, decided = phash_cand, "near"
            how.insert(0, msg + f" — 같은 원본의 watermark {wid}로 확인")
        else:
            a = _aligned_watermark(im, origs, tm)
            if a is not None and a[0].get("file") == phash_cand.get("file"):
                src, decided, aligned_z = a[0], "near", a[2]
                how.insert(0, msg + " — 원본에 맞춘 뒤 watermark로 확인")
    if decided is None and src is not None and wid in wm and not exact:
        # TrustMark 식별자만 맞은 경우(위조 가능한 단서) → 그 원본에 맞춰 내용도 맞는지 확인되면 근거를 'ledger 대조'로 올림
        a = _aligned_watermark(im, origs, tm)
        if a is not None and a[0].get("file") == src.get("file"):
            decided, aligned_z = "aligned", a[2]
            how.insert(0, a[1])
    if not src and not how:
        a = _aligned_watermark(im, origs, tm)
        if a is None:
            return None
        src, how, aligned_z, decided = a[0], [a[1]], a[2], "aligned"
        wid = wid or a[3]
    return {"c2pa": _c2pa(data), "source": (src or {}).get("file"), "caption": (src or {}).get("caption"),
            "how": how, "phash_d": near[0], "tile_d": tile[0], "watermark": wid, "aligned_bdct_z": aligned_z,
            "basis": _basis(exact, decided == "near", decided == "tile", bool(wid), decided == "aligned")}


def _basis(exact, near, tile, wm, aligned):
    """발견의 근거 등급(조사 14 권고 6): ledger 대조는 우리 ledger 기록과 내용이 맞는다는 것,
    watermark는 공개 사본만으로 위조될 수 있어(실험 10) 단서일 뿐이라는 것을 구분해 보고한다."""
    if exact:
        return "ledger 대조: 같은 파일(SHA-256)"
    if near:
        return "ledger 대조: 닮은 그림(perceptual hash)" + (" + watermark 단서" if wm else "")
    if tile:
        return "ledger 대조: 그림 일부(조각 해시)" + (" + watermark 단서" if wm else "")
    if aligned:
        # 정합 단계는 특징점으로 원본과 내용이 맞아야(정합 성공) 그 원본의 그림별 key로 검정한다(실험 18: 무관한 내용에 옮긴 위조 0/80)
        return "ledger 대조: 특징점으로 원본에 정합 + 그 원본의 그림별 key watermark"
    return "watermark 단서만(위조 가능 — ledger 대조 아님)"


def check_media(data, name):
    """음원·영상·문서 파일 → ledger 지문 조회(media_lookup·dfp). 찾으면 결과 dict, 아니면 None."""
    import media_lookup
    import dfp
    suffix = Path(urllib.parse.urlparse(name).path).suffix.lower() or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="E:/envs/tmp") as t:
        t.write(data)
        path = t.name
    try:
        if suffix in (".pdf", ".docx"):
            import fitz
            text = dfp.docx_text(path) if suffix == ".docx" else chr(10).join(p.get_text() for p in fitz.open(path))
            q = dfp.shingles(text)
            best = None
            for lp in LEDGERS:
                for e in ledger.load(lp):
                    if e["record"].get("dfp_bottom"):
                        c = dfp.containment(q, [int(v) for v in e["record"]["dfp_bottom"]])
                        if best is None or c > best[0]:
                            best = (c, e["record"])
            if best and best[0] > dfp.CON_T:
                return {"source": best[1]["file"], "caption": best[1]["caption"], "how": [f"본문 지문 포함도 {best[0]:.2f}"],
                        "basis": "ledger 대조: 문서 지문(MinHash 포함도)", "c2pa": _c2pa_any(data, suffix)}
            return None
        is_video = suffix in (".mp4", ".mov", ".webm", ".mkv")
        r = media_lookup.lookup_video(path) if is_video else media_lookup.lookup(path)
        if not r.get("found"):
            return None
        how = [f"영상 지문 질의측 {r['query_side']:.2f}" + (" (좌우 반전)" if r.get("flipped") else "")] if is_video else               [f"음원 지문 BER {r['ber']:.3f}" + (f" (속도 {r['tempo_factor']}배로 되돌림)" if r.get("tempo_factor") else "")]
        return {"source": r["file"], "caption": r["caption"], "how": how,
                "basis": "ledger 대조: " + ("영상 지문(vPDQ)" if is_video else "음원 지문(Philips)"), "c2pa": _c2pa_any(data, suffix)}
    finally:
        Path(path).unlink(missing_ok=True)


def _c2pa_any(data, suffix):
    from c2pa import Context, Reader
    mime = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".mp4": "video/mp4", ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}.get(suffix)
    if not mime:
        return "없음"
    try:
        with Context() as ctx, Reader(mime, io.BytesIO(data), context=ctx) as r:
            codes = [v["code"] for v in json.loads(r.json()).get("validation_status", [])]
        return "변조" if any("mismatch" in c for c in codes) else "남아 있음"
    except Exception:
        return "없음"


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
        for src in p.media + p.docs:
            url = urllib.parse.urljoin(page, src)
            if not _allowed(url):
                continue
            checked += 1
            m = {}
            hit = check_media(_get(url, m), url)
            if hit:
                found.append({"page": page, "image": url, **hit, "evidence": {"page": page_meta, "image": m}})
            time.sleep(0.5)
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pages": starts, "images_checked": checked, "found": found}


if __name__ == "__main__":
    rep = crawl(sys.argv[1:])
    out = ROOT / "docs" / "assets" / "monitor-report.json"
    text = json.dumps(rep, ensure_ascii=False, indent=1)
    out.write_text(text, encoding="utf-8", newline=chr(10))
    # 보고서 자체의 해시를 ledger에 남긴다 → 다음 비트코인 고정(anchor.py stamp)에 함께 묶여 '이 시각에 이렇게 발견했다'가 고정된다
    if rep["found"]:
        e = ledger.append(LEDGERS[0], {"type": "monitor-evidence", "report_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                       "time": rep["time"], "n_found": len(rep["found"])})
        print("ledger #", e["index"], file=sys.stderr)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
