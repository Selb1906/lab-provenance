"""실험 13: FFDNet·SD-VAE 공격 뒤에도 perceptual hash·ORB 특징점으로 원본을 찾는가.

설계·예측은 notes/개발일지.md "실험 13 설계". 결과는 E:/envs/c2pa-work/exp13/rows.jsonl (이미지마다 저장).
"""
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import ai_attacks  # noqa: E402
import fphash  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402
from exp11_orb_verify import ORB, good_matches  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp13")
CAP = 512


def feats(im):
    g = cv2.cvtColor(__import__("numpy").asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    k, d = ORB.detectAndCompute(g, None)
    return k, d, g.shape


def affine_inliers(q, r):
    gm = good_matches(q, r)
    if gm is None:
        return 0
    _, m = cv2.estimateAffine2D(gm[0], gm[1], method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return int(m.sum()) if m is not None else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    H = [fphash.to_hex(fphash.lab_phash(im)) for _, _, im in items]
    R = [feats(im) for _, _, im in items]
    part = OUT / "rows.jsonl"
    done = {json.loads(l)["i"] for l in part.read_text(encoding="utf-8").splitlines() if l} if part.exists() else set()
    for i, (name, kind, im) in enumerate(items):
        if i in done:
            continue
        s = min(1, CAP / max(im.size))
        small = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        row = {"i": i, "name": name, "kind": kind}
        for atk, fn in [("ffdnet25", ai_attacks.ffdnet), ("sdvae", ai_attacks.vae_regen)]:
            q = fn(small)
            hq = fphash.to_hex(fphash.lab_phash(q))
            fq = feats(q)
            row[atk] = {"phash": [fphash.distance(hq, h) for h in H], "orb": [affine_inliers(fq, r) for r in R]}
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{i + 1}/{len(items)} {name}", flush=True)


if __name__ == "__main__":
    main()
