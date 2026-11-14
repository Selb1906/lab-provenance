"""실험 22: 문서 텍스트 MinHash로 다시 저장·발췌·단어 수정·문단 섞기 뒤에도 원본 문서를 찾는가.

설계·예측은 notes/개발일지.md "실험 22 설계". 결과는 E:/envs/c2pa-work/exp22/summary.json.
"""
import json
import random
import sys
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import dfp  # noqa: E402

PDF = Path("E:/envs/c2pa-work/exp16/pdf")
OUT = Path("E:/envs/c2pa-work/exp22")


def text_of(path):
    with fitz.open(path) as d:
        return "\n".join(p.get_text() for p in d)


def resave(path, out):
    with fitz.open(path) as d:
        d.save(out, garbage=4, deflate=True, clean=True)  # 객체 정리·압축 → 바이트는 바뀌고 본문은 그대로
    return text_of(out)


def excerpt(text, frac, rng):
    w = text.split()
    n = max(50, int(len(w) * frac))
    s = rng.randint(0, max(0, len(w) - n))
    return " ".join(w[s:s + n])


def replace_words(text, frac, rng, vocab):
    w = text.split()
    for i in rng.sample(range(len(w)), int(len(w) * frac)):
        w[i] = rng.choice(vocab)
    return " ".join(w)


def shuffle_paras(text, rng):
    p = [x for x in text.split("\n\n") if x.strip()] or text.split("\n")
    rng.shuffle(p)
    return "\n\n".join(p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF.glob("*.pdf"))
    texts = [text_of(p) for p in pdfs]
    keep = [i for i, t in enumerate(texts) if len(dfp.words(t)) > 2000]  # 본문이 거의 없는(스캔 등) PDF 제외
    pdfs, texts = [pdfs[i] for i in keep], [texts[i] for i in keep]
    SH = [dfp.shingles(t) for t in texts]
    SIG = [dfp.signature(s) for s in SH]
    BK = [dfp.bottom_k(s) for s in SH]
    vocab = sorted({w for t in texts[:5] for w in dfp.words(t)})
    print("docs", len(pdfs), flush=True)
    rows = []
    for i, p in enumerate(pdfs):
        rng = random.Random(i)
        qs = {"resave": resave(p, OUT / "tmp.pdf"), "excerpt50": excerpt(texts[i], 0.5, rng), "excerpt20": excerpt(texts[i], 0.2, rng),
              "replace10": replace_words(texts[i], 0.10, rng, vocab), "shuffle": shuffle_paras(texts[i], rng)}
        for k, q in qs.items():
            qsh = dfp.shingles(q)
            qsig = dfp.signature(qsh)
            rows.append({"i": i, "kind": k, "jac": [dfp.jaccard(qsig, s) for s in SIG], "con": [dfp.containment(qsh, b) for b in BK]})
        print(i, flush=True)
    res = {"n_docs": len(pdfs), "thresholds": {}, "found": {}}
    for m in ["jac", "con"]:
        neg = np.array([max(v for j, v in enumerate(r[m]) if j != r["i"]) for r in rows])
        t = float(np.quantile(neg, 0.99))
        res["thresholds"][m] = {"t": t, "neg_max": float(neg.max())}
        for k in ["resave", "excerpt50", "excerpt20", "replace10", "shuffle"]:
            rs = [r for r in rows if r["kind"] == k]
            res["found"][f"{m}/{k}"] = {"rate": float(np.mean([r[m][r["i"]] > t and r[m][r["i"]] == max(r[m]) for r in rs])),
                                        "self_min": float(min(r[m][r["i"]] for r in rs))}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
