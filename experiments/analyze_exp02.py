"""실험 2 결과 요약: 방식 × 콘텐츠 그룹별 화질, 공격 없을 때 검출, 원본 오탐, 공격별 검출률."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path("E:/envs/c2pa-work/exp02")
ATK = ["rotation", "resizedcrop", "erasing", "brightness", "contrast", "blurring", "noise", "compression"]
KO = {"rotation": "회전", "resizedcrop": "크롭", "erasing": "지우기", "brightness": "밝기", "contrast": "대비",
      "blurring": "흐림", "noise": "잡음", "compression": "JPEG"}
LV = ["0.2", "0.4", "0.6", "0.8", "1.0"]
METHODS = ["blockdct", "dwtDct", "dwtDctSvd", "trustmark"]


def group(kind):
    return "그림" if kind == "figure" else ("사진" if kind == "photo" else "기타")


def main(stage):
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(D.glob(f"*_s{stage}.json"))]
    out = []
    agg = defaultdict(list)
    for r in recs:
        g = group(r["kind"])
        for m, v in r["methods"].items():
            if "error" in v:
                agg[(m, g, "error")].append(1)
                continue
            agg[(m, g, "psnr")].append(v["psnr"])
            agg[(m, g, "clean")].append(bool(v["clean_wm"][1]))
            agg[(m, g, "fp_orig")].append(bool(v["clean_orig"][1]))
            for k, (_, ok) in v["atk"].items():
                agg[(m, g, k)].append(bool(ok))
    groups = ["그림", "사진"]
    out.append(f"표본: {len(recs)}장 (" + ", ".join(f"{g} {sum(1 for r in recs if group(r['kind']) == g)}" for g in groups + ['기타']) + ")\n")
    out.append("| 방식 | 그룹 | PSNR(dB) | 공격 없이 검출 | 원본 오탐 | 삽입 실패 |")
    out.append("|---|---|---|---|---|---|")
    for m in METHODS:
        for g in groups:
            ps = agg[(m, g, "psnr")]
            out.append(f"| {m} | {g} | {np.mean(ps):.1f} | {np.mean(agg[(m, g, 'clean')]):.2f} | "
                       f"{np.mean(agg[(m, g, 'fp_orig')]):.2f} | {len(agg[(m, g, 'error')])} |" if ps else f"| {m} | {g} | - | - | - | {len(agg[(m, g, 'error')])} |")
    if stage == "1":
        for g in groups:
            out.append(f"\n**{g} — 검출률(강도 0.2 → 1.0)**\n")
            out.append("| 방식 | " + " | ".join(KO[a] for a in ATK) + " |")
            out.append("|---|" + "---|" * len(ATK))
            for m in METHODS:
                cells = []
                for a in ATK:
                    vals = [np.mean(agg[(m, g, f"{a}/{lv}")]) if agg[(m, g, f"{a}/{lv}")] else float("nan") for lv in LV]
                    cells.append("/".join("-" if np.isnan(x) else f"{x:.2f}".lstrip("0") for x in vals))
                out.append(f"| {m} | " + " | ".join(cells) + " |")
    else:
        out.append("\n| 방식 | 그룹 | FFDNet σ25 검출 | FFDNet 원본 오탐 | SD-VAE 검출 | SD-VAE 원본 오탐 |")
        out.append("|---|---|---|---|---|---|")
        for m in METHODS:
            for g in groups:
                c = [agg[(m, g, k)] for k in ["ffdnet25", "ffdnet25+orig", "sdvae", "sdvae+orig"]]
                out.append(f"| {m} | {g} | " + " | ".join(f"{np.mean(x):.2f}" if x else "-" for x in c) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1")
