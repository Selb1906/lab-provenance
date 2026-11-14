"""실험 14: SSCD(sscd_disc_mixup) embedding을 원본 후보 생성기로 — 상위 1·5개 적중률과 query 단위 1% threshold 검출률.

방법은 notes/research/10_다음실험_준비.md 2절 그대로. 설계·예측은 notes/개발일지.md "실험 14 설계".
결과는 E:/envs/c2pa-work/exp14/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
from exp01_fingerprint import LEVELS, load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp14")
torch.set_num_threads(4)  # 조사 10 2.3절. 노트북 20스레드 중 일부만 써서 다른 작업과 공존
MODEL = torch.jit.load("E:/envs/c2pa-work/models/sscd/sscd_disc_mixup.torchscript.pt").eval()
PREP = transforms.Compose([transforms.Resize(288), transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def tiles(im):
    w, h = im.size
    boxes = [(0, 0, w, h), (0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h), (w // 2, h // 2, w, h),
             (w // 4, h // 4, 3 * w // 4, 3 * h // 4)]
    return [im.crop(b) for b in boxes]


@torch.no_grad()
def embed(ims):
    out = []
    for im in ims:  # 크기가 제각각이라 한 장씩
        e = MODEL(PREP(im.convert("RGB")).unsqueeze(0))
        out.append(torch.nn.functional.normalize(e, dim=1).numpy()[0])
    return np.stack(out).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    n = len(items)
    REF = np.stack([embed(tiles(im)) for _, _, im in items])  # (n, 6, 512)
    print("refs done", flush=True)
    rows = []
    for i, (name, kind, im) in enumerate(items):
        for atk in ["resizedcrop", "rotation"]:
            for lv in LEVELS:
                q = attacks.apply(im, atk, lv, seed=i)
                q1 = embed([q])            # (1, 512)
                q6 = embed(tiles(q))       # (6, 512)
                s_whole = np.einsum("qd,ntd->nqt", q1, REF).max(axis=(1, 2))
                s_tiles = np.einsum("qd,ntd->nqt", q6, REF).max(axis=(1, 2))
                rows.append((i, kind, atk, lv, {"whole": s_whole.tolist(), "tiles": s_tiles.tolist()}))
        print(f"{i + 1}/{n} {name}", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    res = {"thresholds": {}, "tpr": {}, "top1": {}, "top5": {}}
    for v in ["whole", "tiles"]:
        neg = np.array([max(s for j, s in enumerate(r[4][v]) if j != r[0]) for r in rows])
        t = float(np.quantile(neg, 0.99))  # 이 값보다 큰 음성이 1% 이하
        res["thresholds"][v] = {"t": t, "neg_max": float(neg.max())}
        for g in ["figure", "photo"]:
            for atk in ["resizedcrop", "rotation"]:
                rs = [r for r in rows if r[1] == g and r[2] == atk]
                rank = [int((np.array(r[4][v]) > r[4][v][r[0]]).sum()) for r in rs]
                res["top1"][f"{v}/{g}/{atk}"] = float(np.mean([k == 0 for k in rank]))
                res["top5"][f"{v}/{g}/{atk}"] = float(np.mean([k < 5 for k in rank]))
                res["tpr"][f"{v}/{g}/{atk}"] = float(np.mean([r[4][v][r[0]] > t and k == 0 for r, k in zip(rs, rank)]))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
