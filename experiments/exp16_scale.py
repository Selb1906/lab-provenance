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
    # 첫 실행(11-14 17:50)은 방해 그림 502장을 한꺼번에 메모리에 올렸다가 시스템 메모리 부족으로 중지됨
    # → 방해 그림은 경로만 들고 필요할 때 한 장씩 연다. embedding은 50장마다 중간 저장
    dis_paths = sorted((OUT / "distractors").glob("*.png"))
    N_ALL = n + len(dis_paths)

    def ref_image(j):
        return items[j][2] if j < n else Image.open(dis_paths[j - n]).convert("RGB")

    print("ledger", n, "+ distractors", len(dis_paths), flush=True)
    emb_f = OUT / "ref_emb.npy"
    REF = np.load(emb_f) if emb_f.exists() else np.zeros((0, 6, 512), np.float32)
    for j in range(len(REF), N_ALL):
        REF = np.concatenate([REF, embed(tiles(ref_image(j)))[None]], 0)
        if (j + 1) % 50 == 0 or j + 1 == N_ALL:
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
                        ORBF[j] = feats(ref_image(j))
                    orb[j] = max(inliers(qq, ORBF[j], "B") for qq in Q)
                row = {"i": i, "kind": kind, "atk": atk, "lv": lv, "rank_small": rank_small, "rank_big": rank_big,
                       "top": top, "orb": {str(k): v for k, v in orb.items()}}
                with part.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        print(f"{i + 1}/{n} {name}", flush=True)


if __name__ == "__main__":
    main()
