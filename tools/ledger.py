"""방식 1: 단순 해시체인 원장 (허가형·폐쇄형, 연구실 하나가 운영).

발행한 파일마다 한 줄을 덧붙인다. 각 줄은 앞 줄의 해시를 품고 있어 중간을 고치면 뒤가 전부 어긋난다.
원장 한 줄이 파일 밖에 남기는 것: 파일 해시(SHA-256), 지각 해시(pHash), 워터마크 키, C2PA 매니페스트 식별자.
→ 증명서가 지워진 파일도 워터마크나 지각 해시로 이 줄을 찾아 원래 증명서로 돌아갈 수 있다.

한계(이 방식을 고르면 생기는 문제)는 개발일지에 적는다: 운영자 한 명이 원장 전체를 다시 쓰면 막을 수 없다.
"""
import hashlib
import json
import time
from pathlib import Path

GENESIS = "0" * 64


def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path, record):
    chain = load(path)
    prev = chain[-1]["entry_hash"] if chain else GENESIS
    body = {"index": len(chain), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prev": prev, "record": record}
    body["entry_hash"] = _h({k: v for k, v in body.items()})
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(body, ensure_ascii=False) + "\n")
    return body


def verify(path):
    """처음부터 끝까지 다시 계산해 어긋나는 첫 위치를 돌려준다. 모두 맞으면 None."""
    prev = GENESIS
    for e in load(path):
        body = {k: v for k, v in e.items() if k != "entry_hash"}
        if e["prev"] != prev or _h(body) != e["entry_hash"]:
            return e["index"]
        prev = e["entry_hash"]
    return None


def head(path):
    chain = load(path)
    return chain[-1]["entry_hash"] if chain else GENESIS


def lookup(path, **kv):
    """record 안의 필드로 찾는다. 예: lookup(p, sha256=...)"""
    return [e for e in load(path) if all(e["record"].get(k) == v for k, v in kv.items())]
