"""실험 10: 키 없이 공개된 그림들로 블록 DCT 워터마크 패턴을 추정해 위조·제거할 수 있나 (평균화 공격).

설계·예측은 notes/개발일지.md "실험 10 설계". 결과는 E:/envs/c2pa-work/exp10/.
공격자는 코드는 알고 키는 모른다. 키는 평가(진짜 검출 점수)에만 쓴다.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
import wm_blockdct as W  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp10")
KEY = int(Path("E:/envs/c2pa-site-keys/wm_key.txt").read_text().strip())
Z_T = 2.326  # 오탐 1% (z ~ N(0,1), 실험 2와 같은 기준)


def ycc(img):
    return np.asarray(img.convert("YCbCr"), dtype=np.float64)


def block_dct(y):
    v, h, w = W._blocks(y)
    return dctn(v, axes=(2, 3), norm="ortho"), h, w


def tex_weight(C):
    ac = C.copy()
    ac[:, :, 0, 0] = 0
    tex = np.sqrt((ac ** 2).sum(axis=(2, 3)) / 63.0)
    wj = np.power(np.maximum(tex, 1e-3), W.GAMMA)
    return tex, wj / np.sqrt(np.mean(wj ** 2))


def estimate(imgs, n_stream):
    """난수열 위치(블록 순번 r*nbc+c)별로 질감으로 나눈 계수를 더한다. 질감이 큰 블록은 내용 계수도 커서
    그대로 더하면 내용이 패턴을 덮는다 → 질감으로 나눠 블록마다 잡음 크기를 맞춘다(실험 6의 '사전 백색화 없음' 교훈)."""
    acc = np.zeros((n_stream, 8, 8))
    for im in imgs:
        C, _, _ = block_dct(ycc(im)[:, :, 0])
        tex, _ = tex_weight(C)
        k = C.shape[0] * C.shape[1]
        acc[:k] += (C / (tex[:, :, None, None] + 1.0)).reshape(k, 8, 8)
    s = np.sign(acc)
    s[:, 0, 0] = 0
    return s


def apply_pattern(img, s_stream, sign=+1.0, scale=1.0):
    """embed()와 같은 식으로, 진짜 반송파 대신 추정 패턴을 더하거나 뺀다."""
    Y = ycc(img)
    C, h, w = block_dct(Y[:, :, 0])
    nbr, nbc = C.shape[:2]
    k = nbr * nbc
    s = np.zeros((k, 8, 8))
    m = min(k, len(s_stream))
    s[:m] = s_stream[:m]
    _, wj = tex_weight(C)
    C = C + sign * scale * (W.ALPHA / np.sqrt(63.0)) * wj[:, :, None, None] * s.reshape(nbr, nbc, 8, 8)
    v2 = idctn(C, axes=(2, 3), norm="ortho")
    y2 = Y[:, :, 0].copy()
    y2[:h, :w] = v2.transpose(0, 2, 1, 3).reshape(h, w)
    Y[:, :, 0] = y2
    return Image.fromarray(np.clip(np.round(Y), 0, 255).astype(np.uint8), "YCbCr").convert("RGB")


def psnr(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(10 * np.log10(255 ** 2 / max(np.mean(d ** 2), 1e-12)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pub = [Image.open(p).convert("RGB") for p in sorted((ROOT / "docs" / "assets" / "signed").glob("*.png"))]
    nblk = [(im.height // 8) * (im.width // 8) for im in pub]
    n_stream = max(nblk)
    # 진짜 반송파 난수열(평가용). carrier(key, 1, n)의 블록 순서 = carrier(key, nbr, nbc)를 펼친 순서인지 먼저 확인
    true = W._carrier(KEY, 1, n_stream)[0]
    chk = W._carrier(KEY, pub[0].height // 8, pub[0].width // 8).reshape(-1, 8, 8)
    assert np.array_equal(chk, true[:len(chk)]), "난수열 순서 가정이 틀림"
    res = {"n_published": len(pub), "genuine_z": [W.score(im, KEY) for im in pub], "by_K": {}}
    print("genuine z", [round(z, 1) for z in res["genuine_z"]], flush=True)

    items = load_set()
    clean = [im for _, kind, im in items if kind in ("figure", "photo")][::3][:20]
    rng = np.random.default_rng(0)
    for K in [1, 3, 9]:
        subsets = [list(range(9))] if K == 9 else [sorted(rng.choice(9, K, replace=False)) for _ in range(3)]
        rows = []
        for sub in subsets:
            s_hat = estimate([pub[i] for i in sub], n_stream)
            mask = s_hat != 0
            cover = np.zeros(n_stream, bool)
            cover[:max(nblk[i] for i in sub)] = True
            m = mask & cover[:, None, None]
            agree = float(np.mean(s_hat[m] == true[m]))
            forg = []
            for im in clean:
                f = apply_pattern(im, s_hat, +1)
                forg.append({"z_clean": W.score(im, KEY), "z_forged": W.score(f, KEY), "psnr": psnr(im, f)})
            rem = []
            for i, im in enumerate(pub):
                for sc in [1.0, 2.0]:
                    r = apply_pattern(im, s_hat, -1, sc)
                    rem.append({"img": i, "in_estimate": i in sub, "scale": sc, "z_before": res["genuine_z"][i],
                                "z_after": W.score(r, KEY), "psnr": psnr(im, r)})
            rows.append({"subset": [int(x) for x in sub], "sign_agree": agree, "forgery": forg, "removal": rem})
            fz = [x["z_forged"] for x in forg]
            print(f"K={K} sub={sub} agree={agree:.4f} forged z median={np.median(fz):.1f} "
                  f"success={np.mean([z > Z_T for z in fz]):.2f}", flush=True)
        res["by_K"][str(K)] = rows
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
