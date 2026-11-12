"""방식 2: 원장의 현재 끝(head) 해시를 OpenTimestamps로 비트코인에 고정한다.

원장 전체가 아니라 '이 시각에 원장이 이 상태였다'는 체크포인트 파일 하나를 만들어 도장을 찍는다.
도장은 무료 공개 캘린더 서버가 모아 비트코인 거래 하나에 넣는다(확정까지 몇 시간).

ots 명령줄 클라이언트는 Windows에서 python-bitcoinlib가 OpenSSL DLL을 못 찾아 실행되지 않아,
opentimestamps 핵심 라이브러리로 캘린더 서버에 직접 요청한다(클라이언트와 같은 nonce 방식).

사용법: python tools/anchor.py stamp | upgrade | info
"""
import io
import os
import sys
import time
from pathlib import Path

from opentimestamps.calendar import RemoteCalendar
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.serialize import StreamDeserializationContext, StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile

import ledger

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "assets" / "ledger.jsonl"
ANCHORS = ROOT / "docs" / "assets" / "anchors"
CALENDARS = ["https://a.pool.opentimestamps.org", "https://b.pool.opentimestamps.org", "https://a.pool.eternitywall.com"]


def stamp():
    if ledger.verify(LEDGER) is not None:
        raise SystemExit("ledger is broken; refusing to anchor")
    n = len(ledger.load(LEDGER))
    head = ledger.head(LEDGER)
    ANCHORS.mkdir(parents=True, exist_ok=True)
    cp = ANCHORS / f"ledger-{n:04d}-{head[:12]}.txt"
    if (cp.with_name(cp.name + ".ots")).exists():
        print("already anchored:", cp.name)
        return
    cp.write_text(f"lab-provenance ledger checkpoint\nentries: {n}\nhead: {head}\ncreated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n",
                  encoding="utf-8", newline="\n")
    d = DetachedTimestampFile.from_fd(OpSHA256(), io.BytesIO(cp.read_bytes()))
    merkle = d.timestamp.ops.add(OpAppend(os.urandom(16))).ops.add(OpSHA256())
    ok = 0
    for url in CALENDARS:
        try:
            merkle.merge(RemoteCalendar(url).submit(merkle.msg, timeout=20))
            ok += 1
        except Exception as e:
            print("calendar failed:", url, type(e).__name__)
    if not ok:
        raise SystemExit("no calendar accepted the stamp")
    with open(cp.with_name(cp.name + ".ots"), "wb") as fd:
        d.serialize(StreamSerializationContext(fd))
    print(f"stamped {cp.name} on {ok} calendars")


def _load(p):
    with open(p, "rb") as fd:
        return DetachedTimestampFile.deserialize(StreamDeserializationContext(fd))


def _attestations(ts):
    return [a for _, a in ts.all_attestations()]


def _pending_nodes(ts):
    """대기 증명을 가진 하위 타임스탬프 객체를 모두 찾는다.
    (all_attestations()는 (메시지 바이트, 증명)을 돌려주므로 merge할 객체를 얻으려면 트리를 직접 따라가야 한다)"""
    for att in ts.attestations:
        if isinstance(att, PendingAttestation):
            yield ts, att
    for sub in ts.ops.values():
        yield from _pending_nodes(sub)


def upgrade():
    """대기 중인 도장을 캘린더에 물어 비트코인 블록 증명으로 바꾼다."""
    for p in sorted(ANCHORS.glob("*.ots")):
        d = _load(p)
        changed = False
        for ts, att in list(_pending_nodes(d.timestamp)):
            try:
                ts.merge(RemoteCalendar(att.uri).get_timestamp(ts.msg, timeout=20))
                changed = True
            except Exception as e:  # 아직 비트코인에 안 올라간 캘린더는 '찾을 수 없음'을 돌려준다
                print(f"  {att.uri}: {type(e).__name__}")
        if changed:
            with open(p, "wb") as fd:
                d.serialize(StreamSerializationContext(fd))
        info_one(p)


def info_one(p):
    atts = _attestations(_load(p).timestamp)
    btc = [a.height for a in atts if isinstance(a, BitcoinBlockHeaderAttestation)]
    pend = [a.uri for a in atts if isinstance(a, PendingAttestation)]
    print(f"{p.name}: bitcoin blocks {sorted(set(btc)) or '-'}; pending {len(pend)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stamp"
    if cmd == "stamp":
        stamp()
    elif cmd == "upgrade":
        upgrade()
    else:
        for p in sorted(ANCHORS.glob("*.ots")):
            info_one(p)
