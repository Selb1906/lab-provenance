"""content/site.json 과 그림 목록으로 docs/ 아래 정적 사이트를 만든다.

그림은 웹용으로 줄인 뒤 C2PA 서명해서 docs/assets/signed/ 에 넣는다.
사용법: python tools/build_site.py [--resign]
"""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

from PIL import Image

import fphash
import ledger
import sign as signer_mod

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SIGNED = DOCS / "assets" / "signed"
KEYS = Path("E:/envs/c2pa-site-keys")
FIG_SRC = Path("E:/envs/c2pa-work/figs")
MAX_W = 1600

# 논문별 그림: (논문 id, 추출 파일, 캡션, 콘텐츠 종류)
FIGURES = [
    ("buildings2026", "p13_x251.png", "CVRMSE 분해: 변동계수 요인과 비정형 요인", "산점도"),
    ("buildings2026", "p21_x317.png", "건물 유형별 시간대 오차 패턴", "히트맵"),
    ("buildings2026", "p10_x229.png", "계층적 진단 프레임워크", "구성도"),
    ("kiaebs2026", "p3_x201.png", "연구 프레임워크", "구성도"),
    ("kiaebs2026", "p11_x272.png", "외피 변수별 민감도 지수", "막대그래프"),
    ("energy2024", "p6_x133.png", "수요 예측 과정", "구성도"),
    ("energy2024", "p4_x82.png", "대상 건물 평면", "도면"),
    ("enbuild2024", "p10_x171.png", "BES 모델 구성과 보정 절차", "구성도"),
    ("enbuild2024", "p5_x75.png", "단위 체적 조합 모델", "3D 도식"),
]


def esc(s):
    return html.escape(str(s), quote=True)


def prepare_figures(papers, resign):
    SIGNED.mkdir(parents=True, exist_ok=True)
    tmp = Path("E:/envs/c2pa-work/web")
    tmp.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (pid, fname, caption, kind) in enumerate(FIGURES, 1):
        p = papers[pid]
        name = f"{pid}-{i:02d}.png"
        dst = SIGNED / name
        if resign or not dst.exists():
            im = Image.open(FIG_SRC / pid / fname).convert("RGB")
            if im.width > MAX_W:
                im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
            web = tmp / name
            im.save(web, optimize=True)
            note = f"Figure from {p['venue']} ({p['year']}), doi:{p['doi']}, license {p['license']}. Published by Green Energy Lab, Sangmyung University."
            signer_mod.sign(web, dst, f"{caption} — {p['title'][:60]}", note, KEYS)
        data = dst.read_bytes()
        w, h = Image.open(dst).size
        out.append({
            "file": f"assets/signed/{name}", "paper": pid, "caption": caption, "kind": kind,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "w": w, "h": h,
        })
    (DOCS / "assets" / "figures.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    record_in_ledger(out)
    return out


def record_in_ledger(figs):
    """발행한 그림마다 원장에 한 줄. 이미 있는 파일(같은 SHA-256)은 건너뛴다."""
    from c2pa import Context, Reader
    path = DOCS / "assets" / "ledger.jsonl"
    for f in figs:
        if ledger.lookup(path, sha256=f["sha256"]):
            continue
        p = DOCS / f["file"]
        with Context() as ctx, Reader(str(p), context=ctx) as r:
            manifest_id = json.loads(r.json())["active_manifest"]
        ledger.append(path, {
            "file": f["file"], "sha256": f["sha256"],
            "lab_phash": fphash.to_hex(fphash.lab_phash(Image.open(p))),
            "c2pa_manifest": manifest_id, "caption": f["caption"], "paper": f["paper"],
        })
    # 크롭본을 찾기 위한 조각 해시(실험 5). 본 기록과 별도의 줄로 덧붙인다(이미 고정된 앞 줄은 건드리지 않음)
    for f in figs:
        if ledger.lookup(path, type="tiles", of_sha256=f["sha256"]):
            continue
        ledger.append(path, {"type": "tiles", "of_sha256": f["sha256"], "file": f["file"],
                             "tiles": fphash.tile_hashes(Image.open(DOCS / f["file"]))})
    bad = ledger.verify(path)
    if bad is not None:
        raise SystemExit(f"ledger broken at index {bad}")


NAV = [("index.html", "소개"), ("papers.html", "논문"), ("patents.html", "특허"),
       ("journey.html", "한 장의 여정"), ("provenance.html", "출처증명"), ("verify.html", "검증하기")]

JOURNEY_STEPS = [
    ("0-published.png", "발행", "연구실이 그림을 발행한다. 픽셀 안에 식별자 GEL-0002를 워터마크로 숨기고, C2PA로 서명하고, 원장에 한 줄을 남긴다.",
     "모든 층이 같은 그림을 가리킨다. 이 상태가 기준이다."),
    ("1-messenger.jpg", "메신저로 전달", "누군가 그림을 메신저로 보낸다. 메신저는 사진을 JPEG로 다시 압축한다(품질 74, 공개 벤치마크 WAVES의 가장 약한 압축 단계).",
     "C2PA 증명서가 통째로 사라진다. 그래도 워터마크 식별자와 지각 해시로 원장을 찾아 원래 증명서로 돌아갈 수 있다."),
    ("2-cropped.png", "잘라서 사용", "누군가 그림의 일부(면적 80%)만 잘라 원래 크기로 늘려 쓴다(WAVES 크롭 단계 0.4).",
     "지각 해시는 그림 전체 구도를 보기 때문에 놓친다. 워터마크와 특징점이 대신 찾는다."),
    ("3-screenshot.png", "화면 캡처", "누군가 그림을 브라우저에서 열고 화면 전체(1920×1080)를 캡처한다. 그림 주변에 어두운 배경이 함께 찍힌다.",
     "그림 주변이 섞이면 지각 해시는 놓친다. 워터마크는 위치 검출을 켜면 그림 영역을 찾아 읽는다."),
    ("4-tampered.png", "내용 조작", "누군가 히트맵의 한 칸을 다른 색으로 칠한다. 증명서는 그대로 남겨 둔다.",
     "C2PA만 조작을 잡는다(서명 뒤 바뀜). 워터마크·지각 해시·특징점은 '같은 그림'이라고 답한다 — 이들은 '원래 기록이 어디 있나'를 찾는 도구이지, 변조를 잡는 도구가 아니다."),
    ("5-leaked.jpg", "유출", "같은 그림을 세 곳에 보낼 때 받는 사람마다 다른 식별자를 숨겨 보냈다. 그중 하나가 메신저를 거쳐 퍼진다.",
     "퍼진 사본에서 읽은 식별자 RCV-B002로 누구의 사본이 퍼졌는지 안다. C2PA로는 알 수 없는 정보다."),
]
RECIPIENT_NAMES = {"RCV-A001": "공동연구기관 A", "RCV-B002": "자료 요청자 B", "RCV-C003": "기업 C", "GEL-0002": "연구실 발행본"}


def page(fname, title, body, scripts=""):
    current = ' aria-current="page"'
    nav = "".join(f'<a href="{h}"{current if h == fname else ""}>{t}</a>' for h, t in NAV)
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · 그린에너지연구소</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='4' fill='%231f6b4f'/%3E%3Ccircle cx='8' cy='8' r='3' fill='%23fff'/%3E%3C/svg%3E">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<link rel="stylesheet" href="assets/css/site.css">
</head>
<body>
<header class="top">
  <a class="brand" href="index.html">그린에너지연구소<span>상명대학교</span></a>
  <nav>{nav}</nav>
</header>
<main>
{body}
</main>
<footer>
  <p>이 사이트의 그림은 발행할 때 C2PA로 서명되었습니다. 서명 인증서는 연구실 자체 시험용이며 C2PA 신뢰 목록에 올라 있지 않습니다.</p>
  <p>젠룩 산학협력 프로젝트(2026) 실습 결과물 · <a href="https://github.com/Selb1906/lab-provenance">소스와 개발일지</a></p>
</footer>
{scripts}
</body>
</html>
"""


def seal(fig):
    return (f'<span class="seal" title="SHA-256 {fig["sha256"]}">'
            f'<span class="dot"></span>C2PA 서명됨</span>')


def build_index(site, figs):
    lab, me = site["lab"], site["researcher"]
    hero = next(f for f in figs if f["kind"] == "히트맵")
    topics = "".join(f"<li>{esc(t)}</li>" for t in lab["topics"])
    interests = "".join(f"<li>{esc(t)}</li>" for t in me["interests"])
    body = f"""
<section class="intro">
  <div class="intro-text">
    <p class="eyebrow">{esc(lab['affiliation'])}</p>
    <h1>건물과 전력 설비의 에너지를<br>데이터와 시뮬레이션으로 진단합니다</h1>
    <p class="lead">{esc(lab['name_ko'])}({esc(lab['name_en'])})는 {esc(lab['pi'])} 연구실입니다. 이 사이트에 올린 연구 그림에는 누가 언제 발행했는지 확인할 수 있는 C2PA 증명서가 붙어 있습니다.</p>
    <div class="actions"><a class="btn" href="papers.html">논문 보기</a><a class="link" href="verify.html">그림 검증하기</a></div>
  </div>
  <figure class="intro-fig">
    <img src="{hero['file']}" alt="{esc(hero['caption'])}" width="{hero['w']}" height="{hero['h']}">
    <figcaption>{esc(hero['caption'])} {seal(hero)}</figcaption>
  </figure>
</section>
<section class="two">
  <div><h2>연구 분야</h2><ul class="plain">{topics}</ul></div>
  <div><h2>{esc(me['name'])} <span class="muted">{esc(me['position'])}</span></h2><ul class="plain">{interests}</ul></div>
</section>
"""
    return page("index.html", "소개", body)


def doi_link(p):
    return f'<a href="https://doi.org/{esc(p["doi"])}">doi:{esc(p["doi"])}</a>' if p["doi"] else ""


PUBLISH_NOTE = {
    "full": "오픈액세스(CC BY) — 그림과 전문 링크",
    "figures": "오픈액세스(CC BY-NC) — 출처 표시 후 그림 게재",
    "link": "학회 저작권 — 서지 정보와 DOI만",
    "title": "제목만 게재",
}


def build_papers(site, figs):
    rows = []
    for p in sorted(site["papers"], key=lambda x: -x["year"]):
        pf = [f for f in figs if f["paper"] == p["id"]]
        gallery = "".join(
            # 썸네일을 누르면 서명된 원본 파일이 원래 크기로 열린다 (실험 4: 썸네일 캡처는 어떤 층으로도 못 찾음)
            f'<figure><a href="{f["file"]}" title="원래 크기로 보기 ({f["w"]}×{f["h"]}, C2PA 서명 포함)">'
            f'<img src="{f["file"]}" alt="{esc(f["caption"])}" loading="lazy" width="{f["w"]}" height="{f["h"]}"></a>'
            f'<figcaption>{esc(f["caption"])} <span class="kind">{esc(f["kind"])}</span> {seal(f)}'
            f' <a class="small" href="{f["file"]}" download>원본 받기</a></figcaption></figure>'
            for f in pf)
        rows.append(f"""
<article class="paper">
  <p class="meta"><span class="year">{p['year']}</span>{esc(p['venue'])}</p>
  <h2>{esc(p['title'])}</h2>
  <p class="authors">{esc(p['authors'])}</p>
  <p class="meta">{doi_link(p)} <span class="license">{esc(p['license'])}</span> <span class="muted">{esc(PUBLISH_NOTE[p['publish']])}</span></p>
  {f'<div class="gallery">{gallery}</div>' if gallery else ''}
</article>""")
    body = f"""
<section class="head"><h1>논문</h1>
<p class="lead">논문마다 저작권 조건이 달라 게재 범위를 다르게 정했습니다. 오픈액세스 논문만 그림을 다시 싣고, 나머지는 서지 정보와 DOI만 둡니다.</p></section>
{''.join(rows)}
"""
    return page("papers.html", "논문", body)


def build_patents(site):
    rows = "".join(
        f'<tr><td class="num">{esc(p["number"])}</td><td>{esc(p["title"])}</td><td class="num">{esc(p["registered"])}</td></tr>'
        for p in site["patents"])
    body = f"""
<section class="head"><h1>등록 특허</h1>
<p class="lead">특허권자는 상명대학교 산학협력단, 발명자는 김정욱 외입니다. 공보 원문에는 발명자 주소가 들어 있어 원문 파일은 싣지 않고 서지 정보만 둡니다. 원문은 특허정보검색서비스(KIPRIS)에서 등록번호로 찾을 수 있습니다.</p></section>
<table class="data">
<thead><tr><th>등록번호</th><th>발명의 명칭</th><th>등록일</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p class="muted small">출원 중이거나 공개 전인 특허는 싣지 않습니다.</p>
"""
    return page("patents.html", "특허", body)


def _mark(ok, yes, no):
    return f'<span class="state {"ok" if ok else "bad"} inline"><span class="dot"></span>{yes if ok else no}</span>'


def build_journey():
    res = json.loads((DOCS / "assets" / "journey" / "results.json").read_text(encoding="utf-8"))
    cards = []
    for i, (f, title, what, lesson) in enumerate(JOURNEY_STEPS):
        r = res[f]
        c2 = r["c2pa"]
        c2_cls = "ok" if c2.startswith("유효") else ("bad" if c2 == "변조됨" else "warn")
        wid = r["watermark"]
        ledger_hit = r["sha256_match"] or r["phash_ok"] or (wid == "GEL-0002")
        rows = [
            ("C2PA", f'<span class="state {c2_cls} inline"><span class="dot"></span>{esc(c2)}</span>'),
            ("워터마크", _mark(bool(wid), f"식별자 {esc(wid)} ({esc(RECIPIENT_NAMES.get(wid, ''))})", "못 읽음")),
            ("지각 해시", _mark(r["phash_ok"], f"같은 그림 (거리 {r['phash_d']})", f"놓침 (거리 {r['phash_d']} &gt; 15)")),
            ("특징점(ORB)", _mark(r["orb_ok"], f"같은 그림 (일치 {r['orb_inliers']}개)", f"놓침 (일치 {r['orb_inliers']}개)")),
            ("원장 조회", _mark(ledger_hit, "원래 기록과 증명서를 찾음", "찾지 못함")),
        ]
        table = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
        cards.append(f"""
<article class="step">
  <div class="step-fig"><a href="assets/journey/{f}"><img src="assets/journey/{f}" alt="{esc(title)}" loading="lazy"></a></div>
  <div class="step-body">
    <p class="meta"><span class="year">{i}</span>{esc(title)}</p>
    <p>{esc(what)}</p>
    <table class="data compact">{table}</table>
    <p class="lesson">{esc(lesson)}</p>
  </div>
</article>""")
    tamper = res["ledger_tamper_detected_at"]
    body = f"""
<section class="head"><h1>그림 한 장의 여정</h1>
<p class="lead">연구 그림 한 장이 발행된 뒤 겪을 수 있는 일을 실제로 만들어 보고, 그때마다 각 기술이 무엇을 알려 주는지 검사했습니다. C2PA 하나로는 왜 부족한지, 다른 기술이 어디서 필요한지가 이 순서로 드러납니다.</p></section>
{''.join(cards)}
<section>
  <h2>원장을 몰래 고치면</h2>
  <p>연구실 원장의 한 줄(워터마크 식별자)을 고친 사본을 검증하면 {tamper}번 줄에서 바로 어긋남이 드러납니다. 운영자가 원장 전체를 처음부터 다시 쓰는 경우까지 막기 위해, 원장 끝 해시를 비트코인에 고정해 둡니다(<a href="provenance.html">출처증명</a> 참고).</p>
</section>
<section>
  <h2>정리: 어떤 일을 어떤 기술이 막나</h2>
  <table class="data">
    <thead><tr><th>일어난 일</th><th>C2PA</th><th>워터마크</th><th>지각 해시</th><th>특징점</th><th>원장</th></tr></thead>
    <tbody>
      <tr><td>그대로</td><td>확인</td><td>확인</td><td>확인</td><td>확인</td><td>기준</td></tr>
      <tr><td>재압축(증명서 삭제)</td><td class="muted">사라짐</td><td>찾음</td><td>찾음</td><td>찾음</td><td>원래 증명서로 연결</td></tr>
      <tr><td>잘라서 사용</td><td class="muted">사라짐</td><td>찾음</td><td class="muted">놓침</td><td>찾음</td><td>원래 증명서로 연결</td></tr>
      <tr><td>화면 캡처</td><td class="muted">사라짐</td><td>찾음(위치 검출)</td><td class="muted">놓침</td><td>찾음</td><td>원래 증명서로 연결</td></tr>
      <tr><td>내용 조작</td><td><strong>조작 탐지</strong></td><td class="muted">같다고 답함</td><td class="muted">같다고 답함</td><td class="muted">같다고 답함</td><td>원래 기록과 비교 가능</td></tr>
      <tr><td>유출</td><td class="muted">알 수 없음</td><td><strong>유출자 식별</strong></td><td class="muted">알 수 없음</td><td class="muted">알 수 없음</td><td>받는 사람 대응표 보관</td></tr>
      <tr><td>원장 조작</td><td class="muted">해당 없음</td><td class="muted">해당 없음</td><td class="muted">해당 없음</td><td class="muted">해당 없음</td><td><strong>해시체인 + 비트코인 고정</strong></td></tr>
    </tbody>
  </table>
  <p>C2PA는 "서명 뒤 바뀌었는가"에는 가장 정확하게 답하지만, 증명서가 지워지는 순간 아무것도 말하지 못합니다. 워터마크·지각 해시·특징점은 증명서가 지워진 뒤 "원래 기록이 어디 있나"를 찾아 주고, 원장은 그 기록을 둘 곳과 그 기록이 고쳐지지 않았다는 증거를 줍니다.</p>
  <p class="muted small">이 시연의 그림(히트맵)은 색과 질감이 많아 워터마크에 유리합니다. 흰 배경이 대부분인 그래프·도식에서는 워터마크와 특징점이 크게 약해집니다(<a href="provenance.html">출처증명</a>의 실험 결과 참고). 위 표는 이 그림 한 장의 결과입니다.</p>
</section>
"""
    return page("journey.html", "한 장의 여정", body)


def build_provenance():
    body = (ROOT / "content" / "provenance.html").read_text(encoding="utf-8")
    return page("provenance.html", "출처증명", body)


def build_verify(figs):
    opts = "".join(f'<option value="{f["file"]}">{esc(f["caption"])}</option>' for f in figs)
    opts += (
        '<option value="assets/samples/kiaebs2026-05-resaved.png">[변형] 다시 저장한 그림 — 증명서가 지워짐</option>'
        '<option value="assets/samples/kiaebs2026-05-tampered.png">[변형] 픽셀 1바이트를 바꾼 그림 — 증명서는 남음</option><option value="assets/samples/kiaebs2026-05-cropped.png">[변형] 일부만 잘라 쓴 그림 — 조각으로 찾음</option>'
    )
    body = f"""
<section class="head"><h1>검증하기</h1>
<p class="lead">파일을 올리면 브라우저 안에서 C2PA 증명서를 읽습니다. 파일은 서버로 보내지 않습니다.</p></section>
<section class="verify">
  <div class="panel">
    <label class="drop" for="file"><input id="file" type="file" accept="image/*"><span>이미지 파일을 고르거나 끌어다 놓기</span></label>
    <p class="muted small">또는 이 사이트의 그림으로 해 보기</p>
    <div class="row"><select id="sample">{opts}</select><button id="useSample" class="btn">불러오기</button></div>
    <label class="check"><input id="trustLab" type="checkbox"> 연구실 루트 인증서를 신뢰 목록에 넣기</label>
    <p class="muted small">이 칸을 켜면 같은 파일이 '신뢰 목록에 없는 서명자'에서 '신뢰된 서명자'로 바뀝니다. 신뢰 목록에 오른다는 것이 무엇인지 보여 주는 실험입니다.</p>
  </div>
  <div class="panel result" id="result"><p class="muted">아직 고른 파일이 없습니다.</p></div>
</section>
"""
    scripts = '<script type="module" src="assets/js/verify.js"></script>'
    return page("verify.html", "검증하기", body, scripts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resign", action="store_true")
    a = ap.parse_args()
    site = json.loads((ROOT / "content" / "site.json").read_text(encoding="utf-8"))
    papers = {p["id"]: p for p in site["papers"]}
    figs = prepare_figures(papers, a.resign)
    trust = DOCS / "assets" / "trust"
    trust.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(KEYS / "root.pem", trust / "lab-root.pem")
    pages = {
        "index.html": build_index(site, figs),
        "papers.html": build_papers(site, figs),
        "patents.html": build_patents(site),
        "journey.html": build_journey(),
        "provenance.html": build_provenance(),
        "verify.html": build_verify(figs),
    }
    for name, text in pages.items():
        (DOCS / name).write_text(text, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"built {len(pages)} pages, {len(figs)} signed figures")


if __name__ == "__main__":
    main()
