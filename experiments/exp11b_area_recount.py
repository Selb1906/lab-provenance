"""실험 11b 다시 세기: crop query를 '실제 남은 면적'별로, 서로 다른 crop만 센다(조사 14 — WAVES crop 단계가 겹침).

실제 면적 = side² / (w·h), side = min(round(sqrt(s·w·h)), w, h) (tools/attacks._square_params와 같은 식).
같은 그림·같은 side면 seed가 같아 위치도 같으므로 같은 crop으로 보고 한 번만 센다.
판정은 exp11b와 같음: B(아핀+눌러 보기) 점수 ≥ threshold 96 이고 자기 자신이 최고점.
결과: experiments/results/exp11b_area.json. 모델 없음(행 파일과 그림 크기만 사용).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import attacks  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402

ROWS = Path("E:/envs/c2pa-work/exp11b/rows.jsonl")


def main():
    t = json.loads(Path("E:/envs/c2pa-work/exp11b/summary.json").read_text(encoding="utf-8"))["thresholds"]["B"]["t"]
    sizes = [im.size for _, _, im in load_set()]
    seen, bins = set(), {"<0.55": [], "0.55-0.7": [], ">=0.7": []}
    for line in ROWS.read_text(encoding="utf-8").splitlines():
        i, kind, atk, lv, sc = json.loads(line)
        if kind != "figure" or atk != "resizedcrop":
            continue
        w, h = sizes[i]
        side = max(1, min(int(round((attacks.absolute(atk, lv) * w * h) ** 0.5)), w, h))
        if (i, side) in seen:
            continue
        seen.add((i, side))
        a = side * side / (w * h)
        ok = sc["B"][i] >= t and sc["B"][i] == max(sc["B"])
        bins["<0.55" if a < 0.55 else "0.55-0.7" if a < 0.7 else ">=0.7"].append(ok)
    out = {"threshold_B": t, "distinct_crops": len(seen),
           "by_real_area": {k: {"n": len(v), "found": sum(v), "rate": round(sum(v) / len(v), 3) if v else None} for k, v in bins.items()}}
    (HERE / "results" / "exp11b_area.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
