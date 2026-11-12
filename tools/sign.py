"""파일에 C2PA 매니페스트를 붙여 서명한다.

사용법: python tools/sign.py <입력> <출력> --title "제목" [--keys E:/envs/c2pa-site-keys] [--no-tsa]
"""
import argparse
import json
import mimetypes
from pathlib import Path

from c2pa import Builder, C2paSignerInfo, C2paSigningAlg, Context, Reader, Signer

GENERATOR = {"name": "lab-provenance", "version": "0.1.0"}
TSA_URL = b"http://timestamp.digicert.com"


def make_signer(keys_dir, use_tsa):
    keys = Path(keys_dir)
    info = C2paSignerInfo(
        alg=C2paSigningAlg.ES256,
        sign_cert=(keys / "chain.pem").read_bytes(),
        private_key=(keys / "signer_key.pem").read_bytes(),
        ta_url=TSA_URL if use_tsa else None,
    )
    return Signer.from_info(info)


def manifest(title, source_note):
    return {
        "claim_generator_info": [GENERATOR],
        "title": title,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    # C2PA 규칙: 첫 행동은 created 또는 opened 여야 한다 (published만 쓰면 거부됨)
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCreation",
                        },
                        {
                            "action": "c2pa.published",
                            "description": source_note,
                        },
                    ]
                },
            }
        ],
    }


def sign(src, dst, title, source_note, keys_dir, use_tsa=True):
    mime = mimetypes.guess_type(str(src))[0]
    with Context() as ctx, make_signer(keys_dir, use_tsa) as signer:
        with Builder(json.dumps(manifest(title, source_note)), ctx) as builder:
            with open(src, "rb") as s, open(dst, "w+b") as d:
                builder.sign(signer, mime, s, d)
        with Reader(str(dst), context=ctx) as reader:
            return json.loads(reader.json())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--title", required=True)
    ap.add_argument("--note", default="Published by Green Energy Lab, Sangmyung University")
    ap.add_argument("--keys", default="E:/envs/c2pa-site-keys")
    ap.add_argument("--no-tsa", action="store_true")
    a = ap.parse_args()
    store = sign(a.src, a.dst, a.title, a.note, a.keys, not a.no_tsa)
    active = store["manifests"][store["active_manifest"]]
    print(json.dumps({"title": active.get("title"),
                      "signature_info": active.get("signature_info"),
                      "validation_state": store.get("validation_state"),
                      "validation_status": store.get("validation_status")}, ensure_ascii=False, indent=2))
