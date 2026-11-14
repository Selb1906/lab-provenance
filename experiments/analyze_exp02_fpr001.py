"""실험 2 재집계: watermark 판정 기준을 false positive 0.1%로 통일(조사 04 추천 3). 저장된 점수만 사용, 새 실행 없음.

- block DCT: z > Φ⁻¹(0.999) = 3.090 (기존 2.326)
- DWT-DCT·DWT-DCT-SVD: 32비트 중 맞은 수 ≥ k, k = 무작위 비트가 k개 이상 맞을 확률 ≤ 0.1%인 최소값(이항분포)
- TrustMark: ECC 뒤 8글자 정확 일치라 우연 일치 확률이 이미 0.1%보다 훨씬 작음 → 기준 변화 없음(설명으로 처리)
- 화질 맞춤 추정: block DCT PSNR이 TrustMark보다 약 2dB 높음(세게 심음). z는 심는 세기(alpha)에 비례하므로 2dB 낮추면
  z × 10^(-2/20) ≈ 0.794로 추정(반올림·포화가 있는 공격에서는 근사) — 재실행하지 않은 추정치로 표시.
"""
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

D = Path("E:/envs/c2pa-work/exp02")
ATK = ["rotation", "resizedcrop", "erasing", "brightness", "contrast", "blurring", "noise", "compression"]
LV = ["0.2", "0.4", "0.6", "0.8", "1.0"]
Z1, Z01 = 2.326, 3.090
SCALE = 10 ** (-2 / 20)


def k_for(n, fpr):
    return next(k for k in range(n + 1) if sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n <= fpr)


def main():
    k1, k01 = k_for(32, 0.01), k_for(32, 0.001)
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(D.glob("*_s1.json"))]
    agg = defaultdict(list)
    for r in recs:
        g = "figure" if r["kind"] == "figure" else ("photo" if r["kind"] == "photo" else "other")
        for m, v in r["methods"].items():
            if "error" in v:
                continue
            pairs = [("clean", v["clean_wm"][0]), ("fp_orig", v["clean_orig"][0])] + [(k, s) for k, (s, _) in v["atk"].items()]
            for key, s in pairs:
                if m == "blockdct":
                    agg[(m + "@1%", g, key)].append(s > Z1)
                    agg[(m + "@0.1%", g, key)].append(s > Z01)
                    agg[(m + "@0.1%,-2dB(추정)", g, key)].append(s * SCALE > Z01)
                elif m.startswith("dwt"):
                    agg[(m + "@1%", g, key)].append(s >= k1)
                    agg[(m + "@0.1%", g, key)].append(s >= k01)
                else:
                    agg[(m, g, key)].append(bool(s))
    print(f"bits threshold: 1% → {k1}/32, 0.1% → {k01}/32\n")
    names = ["blockdct@1%", "blockdct@0.1%", "blockdct@0.1%,-2dB(추정)", "dwtDct@1%", "dwtDct@0.1%", "dwtDctSvd@1%", "dwtDctSvd@0.1%", "trustmark"]
    res = {}
    for g in ["figure", "photo"]:
        print(f"## {g} (강도 0.2→1.0)\n| 방식 | 원본 오탐 | " + " | ".join(ATK) + " |\n|---|---|" + "---|" * len(ATK))
        for n in names:
            cells = ["/".join(f"{np.mean(agg[(n, g, f'{a}/{lv}')]):.2f}".lstrip("0") for lv in LV) for a in ATK]
            fp = np.mean(agg[(n, g, "fp_orig")])
            res[f"{g}/{n}"] = {"fp_orig": fp, **{a: [float(np.mean(agg[(n, g, f'{a}/{lv}')])) for lv in LV] for a in ATK}}
            print(f"| {n} | {fp:.2f} | " + " | ".join(cells) + " |")
        print()
    Path("experiments/results/exp02_fpr001.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
