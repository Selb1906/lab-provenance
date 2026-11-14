"""실험 21: vPDQ 방식 영상 fingerprint로 재인코딩·축소·fps 변경·잘라 올리기 뒤에도 원본 영상을 찾는가.

설계·예측은 notes/개발일지.md "실험 21 설계". 결과는 E:/envs/c2pa-work/exp21/summary.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import vfp  # noqa: E402
from exp01_fingerprint import load_set  # noqa: E402

OUT = Path("E:/envs/c2pa-work/exp21")
FF = imageio_ffmpeg.get_ffmpeg_exe()


def make_video(img_path, out, seed):
    """이미지 위를 천천히 이동·확대하는 8초 1280x720 30fps 영상(zoompan)."""
    z = 1.0015 + 0.0005 * (seed % 3)
    subprocess.run([FF, "-y", "-loglevel", "error", "-loop", "1", "-i", str(img_path), "-t", "8",
                    "-vf", f"scale=2560:-2,zoompan=z='min(zoom+{z - 1:.4f},1.4)':x='iw/2-(iw/zoom/2)+{seed * 7}':y='ih/2-(ih/zoom/2)':d=240:s=1280x720:fps=30,format=yuv420p",
                    "-c:v", "libx264", "-crf", "18", str(out)], check=True)


def attack(src, out, kind):
    vf = {"crf35": [], "480p": ["-vf", "scale=-2:480"], "15fps": ["-r", "15"], "sub4s": []}[kind]
    extra = ["-ss", "2", "-t", "4"] if kind == "sub4s" else []
    crf = "35" if kind == "crf35" else "23"
    subprocess.run([FF, "-y", "-loglevel", "error", *extra, "-i", str(src), *vf, "-c:v", "libx264", "-crf", crf, str(out)], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_set()
    picks = [x for x in items if x[1] in ("photo", "figure")][::7][:10]
    srcdir = OUT / "src"
    srcdir.mkdir(exist_ok=True)
    vids, H = [], []
    for k, (name, kind, im) in enumerate(picks):
        ip = srcdir / f"img{k}.png"
        im.convert("RGB").save(ip)
        vp = srcdir / f"v{k}.mp4"
        if not vp.exists():
            make_video(ip, vp, k)
        vids.append(vp)
        H.append(vfp.frame_hashes(vp))
    res = {"genuine": [], "impostor": []}
    for k, vp in enumerate(vids):
        for kind in ["crf35", "480p", "15fps", "sub4s"]:
            ap = OUT / f"v{k}_{kind}.mp4"
            attack(vp, ap, kind)
            hq = vfp.frame_hashes(ap)
            scores = [vfp.match(hq, h) for h in H]
            res["genuine"].append({"v": k, "kind": kind, "self": scores[k],
                                   "others_best_pc": max(s[1] for j, s in enumerate(scores) if j != k),
                                   "others_best_pq": max(s[0] for j, s in enumerate(scores) if j != k)})
        print(k, picks[k][0], flush=True)
    for i in range(len(H)):
        for j in range(len(H)):
            if i != j:
                res["impostor"].append(vfp.match(H[i], H[j]))
    summ = {}
    for kind in ["crf35", "480p", "15fps", "sub4s"]:
        g = [x for x in res["genuine"] if x["kind"] == kind]
        summ[kind] = {"readme_rule(P_c>=0.8)": sum(x["self"][1] >= vfp.P_C for x in g),
                      "query_side>=0.8": sum(x["self"][0] >= 0.8 for x in g), "n": len(g),
                      "self_pq_min": min(x["self"][0] for x in g), "self_pc_min": min(x["self"][1] for x in g),
                      "others_pq_max": max(x["others_best_pq"] for x in g)}
    summ["impostor_pc_max"] = max(s[1] for s in res["impostor"])
    summ["impostor_pq_max"] = max(s[0] for s in res["impostor"])
    res["summary"] = summ
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
