"""실험 18b: 그림별 key로 심은 9장을 평균해 만든 위조 20장을, 정합 없이 각 key_i로 전체 이미지 z를 잰다(그림별 key만의 효과).

설계·예측은 notes/개발일지.md "실험 18b 설계". 결과는 E:/envs/c2pa-work/exp18b/summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
import wm_blockdct as W  # noqa: E402
from build_site import FIG_SRC, FIGURES, MAX_W  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402
from exp10_averaging_attack import apply_pattern, estimate  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp18b")
MASTER = int(Path("E:/envs/c2pa-site-keys/wm_key.txt").read_text().strip())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figs = json.loads((ROOT / "docs" / "assets" / "figures.json").read_text(encoding="utf-8"))
    keys, pub = [], []
    for f, (pid, fname, _, _) in zip(figs, FIGURES):
        im = Image.open(FIG_SRC / pid / fname).convert("RGB")
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        k = W.derive_key(MASTER, f["watermark_id"])
        keys.append(k)
        pub.append(W.embed(im, k))
    n_stream = max((im.height // 8) * (im.width // 8) for im in pub)
    s_hat = estimate(pub, n_stream)
    items = load_set()
    clean = [im.convert("RGB") for _, kind, im in items if kind in ("figure", "photo")][::3][:20]
    rows = []
    for im in clean:
        f = apply_pattern(im, s_hat, +1)
        zs = [W.score(f, k) for k in keys]
        rows.append({"z_each": zs, "z_clean_each": [W.score(im, k) for k in keys]})
    allz = np.array([r["z_each"] for r in rows])          # (20, 9)
    clean_z = np.array([r["z_clean_each"] for r in rows])
    res = {"rows": rows,
           "per_key_pass_rate": float(np.mean(allz > 2.326)), "per_key_z_median": float(np.median(allz)),
           "max_over_keys_pass": int(np.sum(allz.max(axis=1) > 2.326)), "n": len(rows),
           "clean_per_key_pass_rate": float(np.mean(clean_z > 2.326)), "clean_max_over_keys_pass": int(np.sum(clean_z.max(axis=1) > 2.326))}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
