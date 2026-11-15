"""실험 13 요약 재현: experiments/results/exp13_rows.jsonl에서 ORB threshold와 표의 백분율을 다시 계산한다.

처음엔 이 계산을 일회성 명령으로만 해서 저장소에 코드가 없었다(조사 15 지적 9).
- ORB threshold: AI 공격 query 152개의 '자기 아닌 최고 inlier' 분포에서 넘는 비율 ≤1%인 가장 작은 값
  (152개에서 1%는 1.5개라 사실상 음성 최댓값 바로 위가 된다 — 표본 내 threshold)
- 동점(자기 점수 = 남의 최고 점수)은 성공으로 세지 않는 엄격한 셈도 함께(조사 15 지적: 동점을 성공으로 셈)
"""
import json
from pathlib import Path

import numpy as np

ROWS = Path(__file__).resolve().parent / "results" / "exp13_rows.jsonl"


def main():
    R = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l]
    neg = np.array([max(s for j, s in enumerate(r[a]["orb"]) if j != r["i"]) for r in R for a in ["ffdnet25", "sdvae"]])
    T = min(t for t in range(500) if np.mean(neg >= t) <= 0.01)
    print("ORB threshold", T, "negative max", int(neg.max()))
    for a in ["ffdnet25", "sdvae"]:
        for g in ["figure", "photo"]:
            rs = [r for r in R if r["kind"] == g]
            def ph(r, strict):
                d = r[a]["phash"]; s = d[r["i"]]; o = min(x for j, x in enumerate(d) if j != r["i"])
                return s <= 2 and (s < o if strict else s <= o)
            def orb(r, strict):
                d = r[a]["orb"]; s = d[r["i"]]; o = max(x for j, x in enumerate(d) if j != r["i"])
                return s >= T and (s > o if strict else s >= o)
            for strict in (False, True):
                p = np.mean([ph(r, strict) for r in rs]); o = np.mean([orb(r, strict) for r in rs])
                e = np.mean([ph(r, strict) or orb(r, strict) for r in rs])
                print(f"{a} {g} n={len(rs)} {'동점 실패' if strict else '동점 성공'}: phash {p:.3f} orb {o:.3f} either {e:.3f}")


if __name__ == "__main__":
    main()
