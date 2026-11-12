"""능동 포렌식: 받는 사람마다 다른 식별자를 워터마크로 심어 유출자를 추적한다.

같은 그림을 A·B·C에게 보낼 때 각자 다른 8글자 ID(TrustMark 용량 61비트)를 심는다.
유출본에서 ID를 읽으면 누구의 사본인지 안다. 이것이 '포렌식 워터마크(traitor tracing)'.
식별자와 받는 사람의 대응표는 원장에 둔다(워터마크에는 짧은 ID만 들어가므로).
"""
from trustmark import TrustMark

_TM = None


def tm():
    global _TM
    if _TM is None:
        _TM = TrustMark(verbose=False, model_type="Q", loadRemover=False)
    return _TM


def issue(img, recipient_id):
    assert len(recipient_id) <= 8, "TrustMark text mode holds 8 characters"
    return tm().encode(img.convert("RGB"), recipient_id, MODE="text")


def trace(img):
    secret, present, _ = tm().decode(img.convert("RGB"), MODE="text")
    return secret if present else None
