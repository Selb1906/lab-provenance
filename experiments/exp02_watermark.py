"""실험 2: 워터마크 비교 — 일반 처리(WAVES 8종×5단계) 다음 AI 공격(FFDNet σ25, SD-VAE).

설계는 notes/개발일지.md "실험 2 설계". 표본은 실험 1과 같은 이미지에서 고른다.
결과는 E:/envs/c2pa-work/exp02/ 에 이미지별로 이어 쓰며(재개 가능), 끝나면 요약을 만든다.
STAGE=1 이면 일반 처리만, STAGE=2 면 AI 공격만, 기본은 둘 다.
"""
import json
import os
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
import wm_blockdct  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp02")
LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_FIG = int(os.environ.get("N_FIG", "20"))  # CPU 시간 때문에 논문 그림은 일부만. 결정 근거는 일지
STAGE = os.environ.get("STAGE", "12")
KEY = 20261113


def bit_threshold(n, fpr=0.01):
    """무작위 비트(p=0.5)가 k개 이상 맞을 확률이 fpr 이하가 되는 가장 작은 k."""
    for k in range(n + 1):
        tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
        if tail <= fpr:
            return k
    return n + 1


# ---- 방식별 어댑터: embed(img) -> img, detect(img) -> (score, 판정) ----
class BlockDCT:
    name = "blockdct"

    def embed(self, im):
        return wm_blockdct.embed(im, KEY)

    def detect(self, im):
        z = wm_blockdct.score(im, KEY)
        return z, z > 2.326  # 단측 1%


class DwtDct:
    def __init__(self, method):
        from imwatermark import WatermarkDecoder, WatermarkEncoder
        self.method, self.name = method, method
        self.bits = list(np.random.default_rng(KEY).integers(0, 2, 32))
        self.enc = WatermarkEncoder()
        self.enc.set_watermark("bits", self.bits)
        self.Dec = WatermarkDecoder
        self.k = bit_threshold(32)

    def embed(self, im):
        bgr = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
        out = self.enc.encode(bgr, self.method)
        return Image.fromarray(out[:, :, ::-1].copy())

    def detect(self, im):
        bgr = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
        got = self.Dec("bits", 32).decode(bgr, self.method)
        match = int(sum(int(a) == int(b) for a, b in zip(got, self.bits)))
        return match, match >= self.k


class TrustMarkWM:
    name = "trustmark"

    def __init__(self):
        from trustmark import TrustMark
        self.tm = TrustMark(verbose=False, model_type="Q", loadRemover=False)
        # 용량은 ECC 제외 61비트 = 텍스트 8글자. "lab-provenance"는 잘려서 'lab-prov…'로 읽혔다(일지 참고)
        self.secret = "GEL-2026"

    def embed(self, im):
        return self.tm.encode(im.convert("RGB"), self.secret, MODE="text")

    def detect(self, im):
        secret, present, _ = self.tm.decode(im.convert("RGB"), MODE="text")
        ok = bool(present) and secret == self.secret
        return float(ok), ok


def psnr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    mse = ((a - b) ** 2).mean()
    return float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else 99.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    figs = [x for x in items if x[1] == "figure"]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(figs), size=min(N_FIG, len(figs)), replace=False))
    others = [x for x in items if x[1] != "figure"]  # 사진이 앞에 온다(load_set 순서)
    n_other = int(os.environ.get("N_OTHER", str(len(others))))
    sample = [figs[i] for i in pick] + others[:n_other]
    methods = [BlockDCT(), DwtDct("dwtDct"), DwtDct("dwtDctSvd"), TrustMarkWM()]
    ai = None
    if "2" in STAGE:
        import ai_attacks
        ai = ai_attacks.ATTACKS
    for idx, (name, kind, im) in enumerate(sample):
        f = OUT / (name.replace("/", "__") + f"_s{STAGE}.json")
        if f.exists():
            continue
        t0 = time.time()
        rec = {"name": name, "kind": kind, "size": im.size, "methods": {}}
        # 워터마크 없는 원본에 대한 AI 공격 결과는 방식과 무관하므로 한 번만 계산(SD-VAE가 CPU에서 장당 1~2분)
        orig_ai = {an: fn(im) for an, fn in ai.items()} if ai else {}
        for m in methods:
            try:
                wm = m.embed(im)
            except Exception as e:  # 작은 이미지 등에서 삽입 자체가 실패하는 경우도 결과로 남긴다
                rec["methods"][m.name] = {"error": f"{type(e).__name__}: {e}"[:200]}
                continue
            r = {"psnr": psnr(im, wm), "clean_orig": m.detect(im), "clean_wm": m.detect(wm), "atk": {}}
            if "1" in STAGE:
                for atk in attacks.RANGES:
                    for lv in LEVELS:
                        r["atk"][f"{atk}/{lv}"] = m.detect(attacks.apply(wm, atk, lv, seed=idx))
            if ai:
                for an, fn in ai.items():
                    r["atk"][an] = m.detect(fn(wm))
                    r["atk"][an + "+orig"] = m.detect(orig_ai[an])  # 워터마크 없는 이미지에 같은 공격 → 오탐 확인
            rec["methods"][m.name] = r
        f.write_text(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
        print(f"{idx + 1}/{len(sample)} {name} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
