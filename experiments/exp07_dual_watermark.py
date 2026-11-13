"""실험 7: TrustMark + 블록 DCT를 한 그림에 함께 심기. 설계·예측은 notes/개발일지.md "실험 7 설계"."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import ai_attacks  # noqa: E402
import attacks  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402
from exp02_watermark import BlockDCT, TrustMarkWM, psnr  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp07")
CAP = 512
ATK = {
    "jpeg0.2": lambda im, i: attacks.apply(im, "compression", 0.2),
    "jpeg0.6": lambda im, i: attacks.apply(im, "compression", 0.6),
    "crop0.4": lambda im, i: attacks.apply(im, "resizedcrop", 0.4, seed=i),
    "ffdnet25": lambda im, i: ai_attacks.ffdnet(im, 25.0),
    "sdvae": lambda im, i: ai_attacks.vae_regen(im),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    figs = [x for x in items if x[1] == "figure"]
    pick = sorted(np.random.default_rng(0).choice(len(figs), size=5, replace=False))  # 실험 2의 2단계와 같은 고르기
    sample = [figs[i] for i in pick] + [x for x in items if x[1] == "photo"][:5]
    bd, tm = BlockDCT(), TrustMarkWM()
    for i, (name, kind, im) in enumerate(sample):
        f = OUT / (name.replace("/", "__") + ".json")
        if f.exists():
            continue
        r = min(1, CAP / max(im.size))
        im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
        dual = bd.embed(tm.embed(im))
        rec = {"name": name, "kind": kind, "psnr_dual": psnr(im, dual),
               "psnr_tm": psnr(im, tm.embed(im)), "psnr_bd": psnr(im, bd.embed(im)),
               "clean": {"bd": bd.detect(dual)[1], "tm": tm.detect(dual)[1]},
               "orig_fp": {"bd": bd.detect(im)[1], "tm": tm.detect(im)[1]}, "atk": {}}
        for an, fn in ATK.items():
            a = fn(dual, i)
            rec["atk"][an] = {"bd": bool(bd.detect(a)[1]), "tm": bool(tm.detect(a)[1])}
        f.write_text(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
        print(f"{i + 1}/{len(sample)} {name}", flush=True)
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json")) if p.name != "summary.json"]
    summ = {}
    for g in ["figure", "photo"]:
        rs = [x for x in recs if x["kind"] == g]
        summ[g] = {"n": len(rs),
                   "psnr_dual": float(np.mean([x["psnr_dual"] for x in rs])),
                   "psnr_tm": float(np.mean([x["psnr_tm"] for x in rs])),
                   "psnr_bd": float(np.mean([x["psnr_bd"] for x in rs])),
                   "clean_bd": float(np.mean([x["clean"]["bd"] for x in rs])),
                   "clean_tm": float(np.mean([x["clean"]["tm"] for x in rs])),
                   "atk": {an: {"bd": float(np.mean([x["atk"][an]["bd"] for x in rs])),
                                "tm": float(np.mean([x["atk"][an]["tm"] for x in rs])),
                                "either": float(np.mean([x["atk"][an]["bd"] or x["atk"][an]["tm"] for x in rs]))}
                           for an in ATK}}
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
