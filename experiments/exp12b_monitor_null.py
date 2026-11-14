"""실험 12b: 워터마크 없는 무관한 이미지 76장을 monitor 정합 단계(tools/align.py)에 그대로 통과시켜 진짜 key z와 TrustMark 오해독을 잰다.

설계·예측은 notes/개발일지.md "실험 12b 설계". 결과는 E:/envs/c2pa-work/exp12b/ (이미지마다 저장).
"""
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
import align  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp12b")
KEY = int(Path("E:/envs/c2pa-site-keys/wm_key.txt").read_text().strip())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from trustmark import TrustMark
    tm = TrustMark(verbose=False, model_type="Q", loadRemover=False)
    figs = json.loads((ROOT / "docs" / "assets" / "figures.json").read_text(encoding="utf-8"))
    origs = {f["file"]: (Image.open(ROOT / "docs" / f["file"]).convert("RGB"), f) for f in figs}
    refs = [(k, align.orb_features(v[0])) for k, v in origs.items()]
    part = OUT / "rows.jsonl"
    done = {json.loads(l)["name"] for l in part.read_text(encoding="utf-8").splitlines() if l} if part.exists() else set()
    for name, kind, im in load_set():
        if name in done:
            continue
        cand, n_orb = align.candidate(im, refs)
        row = {"name": name, "kind": kind, "candidate": cand, "orb_inliers": n_orb}
        if cand:
            r = align.register(im, origs[cand][0])
            if r:
                back, valid, _, n_sift = r
                s, ok, _ = tm.decode(back, MODE="text")
                row.update({"sift_inliers": n_sift, "valid_frac": float(valid.mean()),
                            "z": align.bdct_z_valid(back, valid, KEY),
                            "tm_match": bool(ok and s == origs[cand][1]["watermark_id"]), "tm_text": s if ok else None})
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, row.get("z"), row.get("tm_match"), flush=True)


if __name__ == "__main__":
    main()
