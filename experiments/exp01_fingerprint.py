"""실험 1: 일반 처리 공격(WAVES 8종 × 5단계)에서 지각 해시가 같은 이미지를 다시 찾는가.

설계와 예측은 notes/개발일지.md "실험 1 설계". 결과는 E:/envs/c2pa-work/exp01/ 에 쓰고 요약을 출력한다.
"""
import json
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import attacks  # noqa: E402

FIG_DIR = Path("E:/envs/c2pa-work/figs")
OUT = Path("E:/envs/c2pa-work/exp01")
SK_PHOTO = ["astronaut", "coffee", "chelsea", "rocket", "hubble_deep_field", "immunohistochemistry", "camera", "coins", "retina"]
SK_OTHER = {"brick": "texture", "grass": "texture", "gravel": "texture", "text": "text", "horse": "binary"}
LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
HASHES = {"aHash": imagehash.average_hash, "pHash": imagehash.phash, "dHash": imagehash.dhash, "wHash": imagehash.whash}
MAX_SIDE = 1024  # 큰 그림(6000px)도 공격 비용을 맞추려고 긴 변을 줄임. 해시는 어차피 8x8~32x32로 줄여 계산


def load_set():
    from skimage import data
    items = []
    for d in sorted(p for p in FIG_DIR.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.png")):
            items.append((f"{d.name}/{f.stem}", "figure", Image.open(f).convert("RGB")))
    for n in SK_PHOTO + list(SK_OTHER):
        a = getattr(data, n)()
        if a.dtype == bool:
            a = (a * 255).astype(np.uint8)
        im = Image.fromarray(a).convert("RGB")
        items.append((f"skimage/{n}", "photo" if n in SK_PHOTO else SK_OTHER[n], im))
    out = []
    for name, kind, im in items:
        if max(im.size) > MAX_SIDE:
            r = MAX_SIDE / max(im.size)
            im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
        out.append((name, kind, im))
    return out


def hashes(im):
    return {k: f(im) for k, f in HASHES.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    print("images:", len(items), {k: sum(1 for _, t, _ in items if t == k) for k in {t for _, t, _ in items}})
    base = [hashes(im) for _, _, im in items]

    # 다른 이미지 쌍의 거리 (오탐 기준)
    impostor = {k: [] for k in HASHES}
    pair_kind = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pair_kind.append((items[i][1], items[j][1]))
            for k in HASHES:
                impostor[k].append(int(base[i][k] - base[j][k]))

    # 같은 이미지 + 공격 후 거리
    genuine = []  # (idx, kind, attack, level, {hash: dist})
    for idx, (name, kind, im) in enumerate(items):
        for atk in attacks.RANGES:
            for lv in LEVELS:
                a = attacks.apply(im, atk, lv, seed=idx)
                h = hashes(a)
                genuine.append((idx, kind, atk, lv, {k: int(base[idx][k] - h[k]) for k in HASHES}))
        print(f"  {idx + 1}/{len(items)} {name}", flush=True)

    # 문턱: 다른 이미지 쌍 중 오탐이 1% 이하가 되는 가장 큰 거리 (거리 <= t 를 '같다'로 판정)
    def threshold(dists, fpr=0.01):
        d = np.sort(np.array(dists))
        best = -1
        for t in range(0, 65):
            if np.mean(d <= t) <= fpr:
                best = t
        return best

    groups = {"all": None, "figure": "figure", "photo": "photo"}
    summary = {"n_images": len(items), "n_impostor_pairs": len(impostor["pHash"]), "thresholds": {}, "tpr": {}}
    for g, sel in groups.items():
        for k in HASHES:
            if sel is None:
                imp = impostor[k]
            else:
                imp = [d for d, pk in zip(impostor[k], pair_kind) if pk == (sel, sel)]
            t = threshold(imp)
            summary["thresholds"][f"{g}/{k}"] = {"t": t, "n_pairs": len(imp), "impostor_median": float(np.median(imp))}
            for atk in attacks.RANGES:
                for lv in LEVELS:
                    rows = [r for r in genuine if r[2] == atk and r[3] == lv and (sel is None or r[1] == sel)]
                    summary["tpr"][f"{g}/{k}/{atk}/{lv}"] = float(np.mean([r[4][k] <= t for r in rows]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    raw = {"items": [(n, t) for n, t, _ in items],
           "genuine": [(i, t, a, l, d) for i, t, a, l, d in genuine],
           "impostor": impostor, "pair_kind": pair_kind}
    (OUT / "raw.json").write_text(json.dumps(raw), encoding="utf-8")
    print(json.dumps(summary["thresholds"], indent=1))


if __name__ == "__main__":
    main()
