"""다른 브라우저(Firefox·WebKit) 확인: 배포 사이트 검증 페이지에서 샘플 6개를 열어 결과 문구와 오류를 기록한다.

설계는 notes/개발일지.md "다른 브라우저 확인 설계". 결과는 experiments/results/browser_compat.json.
사용법: PLAYWRIGHT_BROWSERS_PATH=E:/envs/pw python experiments/browser_compat.py
"""
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://selb1906.github.io/lab-provenance/verify.html"
SAMPLES = ["assets/samples/kiaebs2026-05-resaved.png", "assets/samples/melody-32k.mp3", "assets/samples/melody-slow.mp3",
           "assets/samples/slides-flipped.mp4", "assets/samples/report-excerpt.pdf", "assets/samples/report-edited.docx"]


def run(engine):
    out = {}
    with sync_playwright() as p:
        b = getattr(p, engine).launch()
        page = b.new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)[:150]))
        page.goto(f"{URL}?nc={time.time()}")
        for s in SAMPLES:
            page.evaluate("document.querySelector('#result').innerHTML=''")
            page.select_option("#sample", s)
            page.click("#useSample")
            try:
                page.wait_for_function("[...document.querySelectorAll('#result h2.sub')].some(h => h.textContent.includes('원장'))", timeout=180000)
                states = page.eval_on_selector_all("#result .state", "els => els.map(e => e.textContent.trim().slice(0, 60))")
                detail = page.eval_on_selector_all("#result details dd, #result p.muted.small", "els => els.map(e => e.textContent.trim().slice(0, 90))")
                out[s.split("/")[-1]] = {"states": states, "detail": detail[:2]}
            except Exception as e:  # noqa: BLE001
                out[s.split("/")[-1]] = {"error": str(e)[:150]}
        out["_page_errors"] = errs
        b.close()
    return out


def main():
    res = {}
    for engine in ["firefox", "webkit"]:
        try:
            res[engine] = run(engine)
        except Exception as e:  # noqa: BLE001
            res[engine] = {"launch_error": str(e)[:200]}
        print(engine, json.dumps(res[engine], ensure_ascii=False)[:1500], flush=True)
    Path(__file__).resolve().parent.joinpath("results", "browser_compat.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
