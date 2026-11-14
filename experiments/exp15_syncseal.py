"""실험 15: SyncSeal(동기화 watermark)을 맨 바깥에 심고, 원본 없이 회전·crop을 되돌린 뒤 TrustMark·block DCT를 읽는다.

방법은 notes/research/10_다음실험_준비.md 3절 그대로. 설계·예측은 notes/개발일지.md "실험 15 설계".
결과는 E:/envs/c2pa-work/exp15/rows.jsonl (이미지마다 저장).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
import wm_blockdct as W  # noqa: E402
from exp01_fingerprint import LEVELS, load_set  # noqa: E402
from exp12_register_detect import true_map  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp15")
KEY = int(Path("E:/envs/c2pa-site-keys/wm_key.txt").read_text().strip())
SECRET = "GEL-2026"
torch.set_num_threads(4)
torch.manual_seed(0)
SS = torch.jit.load("E:/envs/c2pa-work/models/syncseal/syncmodel.jit.pt", map_location="cpu").eval()


def to_pil(t):
    return Image.fromarray((t[0].detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def psnr(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(10 * np.log10(255 ** 2 / max(np.mean(d ** 2), 1e-12)))


def sample():
    """실험 2와 같은 고르기: 그림은 seed 0으로 20장, 사진은 전부(9장)."""
    items = load_set()
    figs = [x for x in items if x[1] == "figure"]
    pick = sorted(np.random.default_rng(0).choice(len(figs), size=20, replace=False))
    return [figs[i] for i in pick] + [x for x in items if x[1] == "photo"]


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from trustmark import TrustMark
    tm = TrustMark(verbose=False, model_type="Q", loadRemover=False)
    part = OUT / "rows.jsonl"
    done = {json.loads(l)["name"] for l in part.read_text(encoding="utf-8").splitlines() if l} if part.exists() else set()
    for idx, (name, kind, im) in enumerate(sample()):
        if name in done:
            continue
        im = im.convert("RGB")
        w, h = im.size
        two = W.embed(tm.encode(im, SECRET, MODE="text"), KEY)            # 사이트 발행과 같은 두 겹
        three = to_pil(SS.embed(to_tensor(two).unsqueeze(0))["imgs_w"])     # + SyncSeal 맨 바깥
        row = {"name": name, "kind": kind, "size": [w, h], "psnr_two": psnr(im, two), "psnr_three": psnr(im, three),
               "psnr_sync_only": psnr(two, three), "atk": {}}
        for atk in ["rotation", "resizedcrop"]:
            for lv in LEVELS:
                q = attacks.apply(three, atk, lv, seed=idx)
                s0, ok0, _ = tm.decode(q, MODE="text")
                x = to_tensor(q).unsqueeze(0)
                det = SS.detect(x)
                pts = det["preds_pts"]
                un = to_pil(SS.unwarp(x, pts, (h, w)))
                s1, ok1, _ = tm.decode(un, MODE="text")
                # 모서리 오차: SyncSeal preds_pts는 '질의의 네 모서리(TL·TR·BR·BL)가 원본 좌표에서 있는 자리'(사전 시험에서
                # 반대 규약은 수백 px 어긋나고 이 규약은 수십 px로 맞음). 참값 = 참 변환(질의→원본)으로 질의 모서리를 옮긴 것
                T = true_map(atk, lv, idx, w, h)
                corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], float).T
                tq = T @ corners
                tq = (tq[:2] / tq[2]).T
                p = pts.detach().view(4, 2).numpy()
                pq = np.stack([p[:, 0] * (w - 1) / 2 + (w - 1) / 2, p[:, 1] * (h - 1) / 2 + (h - 1) / 2], 1)
                row["atk"][f"{atk}/{lv}"] = {
                    "tm_direct": bool(ok0 and s0 == SECRET), "tm_sync": bool(ok1 and s1 == SECRET),
                    "bdct_direct": W.score(q, KEY), "bdct_sync": W.score(un, KEY),
                    "corner_err": float(np.mean(np.linalg.norm(pq - tq, axis=1)))}
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        a = row["atk"]
        print(f"{idx + 1} {name} psnr3={row['psnr_three']:.1f} rot45 err={a['rotation/1.0']['corner_err']:.1f} "
              f"tm={a['rotation/1.0']['tm_direct']}->{a['rotation/1.0']['tm_sync']} z={a['rotation/1.0']['bdct_sync']:.1f}", flush=True)


if __name__ == "__main__":
    main()
