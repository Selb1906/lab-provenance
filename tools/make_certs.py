"""연구실 자체 인증서 체인을 만든다: 루트(상위 기관) -> 중간(발급 기관) -> 서명용(개별 인증서).

공인 CA 인증서가 없으므로 C2PA 검증기에서는 "신뢰 목록에 없는 서명자"로 표시된다.
이것은 의도된 결과이며, 젠룩 포털이 '파일럿·미등재' 단계인 것과 같은 상황을 재현한다.

사용법: python tools/make_certs.py <출력 폴더>
개인키가 들어가므로 출력 폴더는 저장소 밖에 둔다.
"""
import datetime as dt
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

ORG = "Green Energy Lab, Sangmyung University (test)"


def name(cn):
    return x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "KR"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORG),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])


def issue(subject_cn, subject_key, issuer_cn, issuer_key, is_ca, path_len, days):
    now = dt.datetime.now(dt.timezone.utc)
    b = (
        x509.CertificateBuilder()
        .subject_name(name(subject_cn))
        .issuer_name(name(issuer_cn))
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=path_len), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(subject_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()), critical=False)
    )
    if is_ca:
        b = b.add_extension(x509.KeyUsage(
            digital_signature=False, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
            encipher_only=False, decipher_only=False), critical=True)
    else:
        # C2PA 서명 인증서 조건: CA가 아니고, digitalSignature, EKU에 emailProtection 등 허용 목적 포함
        b = b.add_extension(x509.KeyUsage(
            digital_signature=True, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=False, crl_sign=False,
            encipher_only=False, decipher_only=False), critical=True)
        b = b.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
    return b.sign(issuer_key, hashes.SHA256())


def pem(cert):
    return cert.public_bytes(serialization.Encoding.PEM)


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root_key, ica_key, ee_key = (ec.generate_private_key(ec.SECP256R1()) for _ in range(3))
    root = issue("Green Energy Lab Test Root CA", root_key, "Green Energy Lab Test Root CA", root_key, True, 1, 3650)
    ica = issue("Green Energy Lab Test Issuing CA", ica_key, "Green Energy Lab Test Root CA", root_key, True, 0, 1825)
    ee = issue("Green Energy Lab Content Signer", ee_key, "Green Energy Lab Test Issuing CA", ica_key, False, None, 365)

    # 서명 시 넘기는 체인: 개별 인증서 + 발급 기관 (루트는 검증하는 쪽이 신뢰 앵커로 가진다)
    (out / "chain.pem").write_bytes(pem(ee) + pem(ica))
    (out / "root.pem").write_bytes(pem(root))
    (out / "signer_key.pem").write_bytes(ee_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    print(f"written to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
