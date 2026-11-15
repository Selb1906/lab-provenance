"""실험 26: 짧은 연속 발췌(50~800단어)에서 문서 지문(포함도)이 원본을 찾는가, 표본 수와 오탐의 관계.

설계·예측은 notes/개발일지.md "실험 26 설계". 결과는 experiments/results/exp26_summary.json.
"""
import json
import random
import sys
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import dfp  # noqa: E402

PDF = Path("E:/envs/c2pa-work/exp16/pdf")
LENGTHS = [50, 100, 200, 400, 800]


def main():
    texts = []
    for p in sorted(PDF.glob("*.pdf")):
        with fitz.open(p) as d:
            t = "\n".join(x.get_text() for x in d)
        if len(dfp.words(t)) > 2000:  # 실험 22와 같은 조건
            texts.append(t)
    BK = [dfp.bottom_k(dfp.shingles(t)) for t in texts]
    rows = []
    for i, t in enumerate(texts):
        w = dfp.words(t)
        rng = random.Random(i)
        for L in LENGTHS:
            for _ in range(3):
                s = rng.randint(0, len(w) - L)
                q = dfp.shingles(" ".join(w[s:s + L]))
                res = [dfp.containment_n(q, b) for b in BK]
                rows.append({"i": i, "L": L, "self": res[i][0], "n": res[i][1],
                             "other_best": max(c for j, (c, _) in enumerate(res) if j != i),
                             "other_n": max(nn for j, (_, nn) in enumerate(res) if j != i)})
    out = {"n_docs": len(texts), "by_length": {}}
    for L in LENGTHS:
        rs = [r for r in rows if r["L"] == L]
        found = [r["self"] > dfp.CON_T and r["self"] > r["other_best"] for r in rs]
        fp = [r["other_best"] > dfp.CON_T for r in rs]
        out["by_length"][L] = {"queries": len(rs), "found": sum(found) / len(rs), "false_hit": sum(fp) / len(rs),
                               "sample_n_median": sorted(r["n"] for r in rs)[len(rs) // 2],
                               "sample_n_min": min(r["n"] for r in rs)}
        print(L, out["by_length"][L], flush=True)
    # 표본 수 구간별 오탐(자기 아닌 문서가 threshold를 넘는 비율)
    bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 10**9)]
    out["false_hit_by_sample_n"] = {}
    for lo, hi in bins:
        rs = [r for r in rows if lo <= r["other_n"] < hi]
        if rs:
            out["false_hit_by_sample_n"][f"{lo}-{hi}"] = {"n": len(rs), "false_hit": sum(r["other_best"] > dfp.CON_T for r in rs) / len(rs)}
    print(json.dumps(out["false_hit_by_sample_n"], indent=1))
    (HERE / "results" / "exp26_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
