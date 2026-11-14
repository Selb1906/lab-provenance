"""실험 16: 원장에 같은 분야 방해 그림을 섞으면 'SSCD 상위 5개 → ORB+눌러 보기 확인'이 버티는가.

설계·예측은 notes/개발일지.md "실험 16 설계". 결과는 E:/envs/c2pa-work/exp16/ (query마다 jsonl 저장).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
from exp01_fingerprint import LEVELS, load_set  # noqa: E402
from exp11b_orb_squash import feats, inliers, squash_versions  # noqa: E402
from exp14_sscd_candidates import embed, tiles  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp16")
K = 5


def main():
    items = load_set()
    n = len(items)
    dis = [Image.open(p).convert("RGB") for p in sorted((OUT / "distractors").glob("*.png"))]
    refs = [im for _, _, im in items] + dis   # 앞 n개가 실험 집합(정답 후보), 뒤가 방해 그림
    print("ledger", n, "+ distractors", len(dis), flush=True)
    emb_f = OUT / "ref_emb.npy"
    if emb_f.exists() and np.load(emb_f).shape[0] == len(refs):
        REF = np.load(emb_f)
    else:
        REF = np.stack([embed(tiles(im)) for im in refs])
        np.save(emb_f, REF)
    print("ref embeddings", REF.shape, flush=True)
    ORBF = {}
    part = OUT / "rows.jsonl"
    done = {(r["i"], r["atk"], r["lv"]) for r in map(json.loads, part.read_text(encoding="utf-8").splitlines())} if part.exists() else set()
    for i, (name, kind, im) in enumerate(items):
        for atk in ["resizedcrop", "rotation"]:
            for lv in LEVELS:
                if (i, atk, lv) in done:
                    continue
                q = attacks.apply(im, atk, lv, seed=i)
                s = np.einsum("qd,ntd->nqt", embed(tiles(q)), REF).max(axis=(1, 2))
                rank_small = int((s[:n] > s[i]).sum())        # 원장 76장만
                rank_big = int((s > s[i]).sum())              # 76 + 방해 그림
                top = [int(j) for j in np.argsort(-s)[:K]]
                Q = squash_versions(q)
                orb = {}
                for j in top:
                    if j not in ORBF:
                        ORBF[j] = feats(refs[j])
                    orb[j] = max(inliers(qq, ORBF[j], "B") for qq in Q)
                row = {"i": i, "kind": kind, "atk": atk, "lv": lv, "rank_small": rank_small, "rank_big": rank_big,
                       "top": top, "orb": {str(k): v for k, v in orb.items()}}
                with part.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        print(f"{i + 1}/{n} {name}", flush=True)


if __name__ == "__main__":
    main()
