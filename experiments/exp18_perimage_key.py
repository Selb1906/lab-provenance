"""실험 18: 그림마다 다른 block DCT key(주 key + 원장 식별자 HMAC)로 평균화·복사 위조를 monitor 파이프라인에서 막는가.

설계·예측은 notes/개발일지.md "실험 18 설계". 결과는 E:/envs/c2pa-work/exp18/summary.json.
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
import align  # noqa: E402
import attacks  # noqa: E402
import wm_blockdct as W  # noqa: E402
from build_site import FIG_SRC, FIGURES, MAX_W  # noqa: E402
from exp01_fingerprint import LEVELS, load_set  # noqa: E402
from exp10_averaging_attack import apply_pattern, estimate  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp18")
MASTER = int(Path("E:/envs/c2pa-site-keys/wm_key.txt").read_text().strip())


def pipeline(im, refs, origs):
    """monitor와 같은 순서: 후보 원본(ORB+눌러 보기) → SIFT 정합 → 그 후보의 key로 유효 블록 z."""
    name, n_orb = align.candidate(im, refs)
    if name is None:
        return None, None
    r = align.register(im, origs[name]["im"])
    if r is None:
        return name, None
    back, valid, _, _ = r
    return name, align.bdct_z_valid(back, valid, origs[name]["key"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figs = json.loads((ROOT / "docs" / "assets" / "figures.json").read_text(encoding="utf-8"))
    origs = {}
    for f, (pid, fname, _, _) in zip(figs, FIGURES):
        im = Image.open(FIG_SRC / pid / fname).convert("RGB")
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        key = W.derive_key(MASTER, f["watermark_id"])
        origs[f["watermark_id"]] = {"im": W.embed(im, key), "key": key}
    ids = list(origs)
    refs = [(k, align.orb_features(v["im"])) for k, v in origs.items()]
    pub = [origs[k]["im"] for k in ids]
    n_stream = max((im.height // 8) * (im.width // 8) for im in pub)
    items = load_set()
    clean = [im.convert("RGB") for _, kind, im in items if kind in ("figure", "photo")][::3][:20]
    res = {"genuine_z_own_key": {k: W.score(v["im"], v["key"]) for k, v in origs.items()}}

    # ① 평균화 공격: 9장 전부로 추정한 패턴을 20장에 위조 → 파이프라인
    s_hat = estimate(pub, n_stream)
    rows = []
    for im in clean:
        f = apply_pattern(im, s_hat, +1)
        cand, z = pipeline(f, refs, origs)
        rows.append({"candidate": cand, "z": z})
    res["averaging_K9"] = rows
    print("averaging done", sum((r["z"] or 0) > align.Z_T for r in rows), "/", len(rows), flush=True)

    # ② 한 장 복사 공격: 그림 i 한 장에서 추정 → 20장 → 파이프라인 (i = 처음 3장)
    res["copy_K1"] = {}
    for i in range(3):
        s1 = estimate([pub[i]], n_stream)
        rows = []
        for im in clean:
            f = apply_pattern(im, s1, +1)
            cand, z = pipeline(f, refs, origs)
            # 공격자에게 가장 유리한 경우: 검증자가 후보로 i를 골랐다고 가정한 z(전체 이미지 z, key_i)
            rows.append({"candidate": cand, "z": z, "z_if_candidate_i": W.score(f, origs[ids[i]]["key"])})
        res["copy_K1"][ids[i]] = rows
        print("copy from", ids[i], "pipeline hits", sum((r["z"] or 0) > align.Z_T for r in rows),
              "| if verifier picked i:", sum(r["z_if_candidate_i"] > align.Z_T for r in rows), flush=True)

    # ③ 정상 검출: crop·회전 뒤 파이프라인
    rows = []
    for k in ids:
        for atk in ["resizedcrop", "rotation"]:
            for lv in LEVELS:
                q = attacks.apply(origs[k]["im"], atk, lv, seed=ids.index(k))
                cand, z = pipeline(q, refs, origs)
                rows.append({"id": k, "atk": atk, "lv": lv, "candidate": cand, "z": z})
    res["genuine_attacked"] = rows
    print("genuine detected", sum(r["candidate"] == r["id"] and (r["z"] or 0) > align.Z_T for r in rows), "/", len(rows), flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
