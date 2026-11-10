"""
Tests for the secp256k1 compact recoverable signature format:
one recovery id byte followed by 32 byte big-endian r and s.
"""

import hashlib

import pytest

from .keys import (
    VerifyingKey,
    SigningKey,
    BadSignatureError,
)
from .curves import SECP256k1, NIST256p, Ed25519
from .util import number_to_string, string_to_number, MalformedSignature
from . import numbertheory


SECEXP = 0x9D61B19DEFFD5A60BA844AF492EC2CC44449C5697B326919703BAC031CAE7F60
MESSAGE = b"a compact recoverable signature test message"
ORDER = SECP256k1.order
P = SECP256k1.curve.p()


@pytest.fixture
def signing_key():
    return SigningKey.from_secret_exponent(
        SECEXP, curve=SECP256k1, hashfunc=hashlib.sha256
    )


def encode_sig(recid, r, s):
    return (
        bytes(bytearray([recid]))
        + number_to_string(r, ORDER)
        + number_to_string(s, ORDER)
    )


def test_sign_recover_round_trip(signing_key):
    sig = signing_key.sign_recoverable(MESSAGE)
    assert len(sig) == 65
    assert sig[0] in (0, 1, 2, 3)

    recovered = VerifyingKey.from_recoverable_signature(
        sig, MESSAGE, hashfunc=hashlib.sha256
    )
    expected = signing_key.get_verifying_key()
    assert recovered.to_string("compressed") == expected.to_string(
        "compressed"
    )
    assert recovered.to_string("raw") == expected.to_string("raw")

    # the recovered key really verifies the message
    raw_rs = sig[1:]
    assert recovered.verify(raw_rs, MESSAGE, hashfunc=hashlib.sha256)


def test_sign_digest_recover_round_trip(signing_key):
    digest = hashlib.sha256(MESSAGE).digest()
    sig = signing_key.sign_digest_recoverable(digest)
    assert len(sig) == 65

    recovered = VerifyingKey.from_recoverable_signature_with_digest(
        sig, digest, hashfunc=hashlib.sha256
    )
    expected = signing_key.get_verifying_key()
    assert recovered.to_string("raw") == expected.to_string("raw")
    assert recovered.verify_digest(sig[1:], digest)


def test_message_and_digest_entry_points_are_distinct(signing_key):
    # hashing must not be applied twice: recovering the digest signature
    # through the message entry point (which hashes the digest again)
    # must not yield the signing key
    digest = hashlib.sha256(MESSAGE).digest()
    sig = signing_key.sign_digest_recoverable(digest)

    recovered = VerifyingKey.from_recoverable_signature_with_digest(
        sig, digest
    )
    assert (
        recovered.to_string("raw")
        == signing_key.get_verifying_key().to_string("raw")
    )

    # a signature over the message itself recovers a different key pair
    other = SigningKey.from_secret_exponent(
        SECEXP + 1, curve=SECP256k1, hashfunc=hashlib.sha256
    )
    other_sig = other.sign_recoverable(MESSAGE)
    other_recovered = VerifyingKey.from_recoverable_signature(
        other_sig, MESSAGE, hashfunc=hashlib.sha256
    )
    assert (
        other_recovered.to_string("raw")
        == other.get_verifying_key().to_string("raw")
    )
    assert (
        other_recovered.to_string("raw") != recovered.to_string("raw")
    )


def test_deterministic_signing(signing_key):
    sig1 = signing_key.sign_recoverable(MESSAGE)
    sig2 = signing_key.sign_recoverable(MESSAGE)
    assert sig1 == sig2


def test_low_s_normalisation(signing_key):
    n = ORDER
    # find a message that produces a high-s signature without normalisation
    high_s_sig = None
    for i in range(100):
        sig = signing_key.sign_recoverable(
            MESSAGE + bytes(bytearray([i]))
        )
        s = string_to_number(sig[33:])
        if s > n // 2:
            high_s_sig = sig
            break
    assert high_s_sig is not None, "no high-s signature found in 100 tries"

    # the unnormalised signature still recovers the correct key
    recovered = VerifyingKey.from_recoverable_signature(
        high_s_sig, MESSAGE + bytes(bytearray([i])), hashfunc=hashlib.sha256
    )
    assert (
        recovered.to_string("raw")
        == signing_key.get_verifying_key().to_string("raw")
    )

    # low_s=True on the same message produces the normalised pair:
    # same r, s' = n - s, recovery id parity bit flipped
    norm_sig = signing_key.sign_recoverable(
        MESSAGE + bytes(bytearray([i])), low_s=True
    )
    assert string_to_number(norm_sig[33:]) <= n // 2
    assert norm_sig[1:33] == high_s_sig[1:33]
    assert (
        string_to_number(norm_sig[33:])
        == n - string_to_number(high_s_sig[33:])
    )
    assert norm_sig[0] == high_s_sig[0] ^ 1

    # the normalised signature recovers the same key, not the mirror key
    norm_recovered = VerifyingKey.from_recoverable_signature(
        norm_sig, MESSAGE + bytes(bytearray([i])), hashfunc=hashlib.sha256
    )
    assert (
        norm_recovered.to_string("raw")
        == signing_key.get_verifying_key().to_string("raw")
    )

    # flipping s without flipping the recovery id recovers a *different*
    # key that also verifies - the two must stay in sync
    desynced = (
        bytes(bytearray([high_s_sig[0]]))
        + high_s_sig[1:33]
        + norm_sig[33:]
    )
    desynced_recovered = VerifyingKey.from_recoverable_signature(
        desynced, MESSAGE + bytes(bytearray([i])), hashfunc=hashlib.sha256
    )
    assert (
        desynced_recovered.to_string("raw")
        != signing_key.get_verifying_key().to_string("raw")
    )


def test_low_s_already_low(signing_key):
    n = ORDER
    for i in range(100):
        msg = MESSAGE + bytes(bytearray([i]))
        sig = signing_key.sign_recoverable(msg)
        if string_to_number(sig[33:]) <= n // 2:
            norm_sig = signing_key.sign_recoverable(msg, low_s=True)
            # already normalised: nothing changes
            assert norm_sig == sig
            break
    else:
        raise AssertionError("no low-s signature found in 100 tries")


def test_both_recovery_id_parities_occur(signing_key):
    seen = set()
    for i in range(50):
        sig = signing_key.sign_recoverable(
            b"parity sample " + bytes(bytearray([i]))
        )
        recid = sig[0]
        # x overflow is cryptographically improbable on secp256k1
        assert recid in (0, 1)
        seen.add(recid)
        recovered = VerifyingKey.from_recoverable_signature(
            sig, b"parity sample " + bytes(bytearray([i])),
            hashfunc=hashlib.sha256,
        )
        assert (
            recovered.to_string("raw")
            == signing_key.get_verifying_key().to_string("raw")
        )
    assert seen == {0, 1}


def _find_overflow_x():
    # x coordinate in [n, p) that lies on the curve and yields a valid
    # (non-zero) r = x - n
    x = ORDER + 1
    while True:
        alpha = (
            pow(x, 3, P)
            + SECP256k1.curve.a() * x
            + SECP256k1.curve.b()
        ) % P
        if numbertheory.jacobi(alpha, P) == 1:
            return x
        x += 1


def test_recovery_id_with_x_overflow():
    # real signatures essentially never need recid 2 or 3 on secp256k1
    # (p - n is tiny compared to p), so build the case synthetically:
    # for any r, s, e and nonce point R with x(R) == r + n, the
    # recovered key verifies the signature by construction
    x = _find_overflow_x()
    r = x - ORDER
    s = 0xDEADBEEF
    digest = hashlib.sha256(b"overflow case").digest()

    keys = []
    for parity in (0, 1):
        sig = encode_sig(2 | parity, r, s)
        vk = VerifyingKey.from_recoverable_signature_with_digest(
            sig, digest
        )
        assert vk.verify_digest(sig[1:], digest)
        keys.append(vk.to_string("raw"))

    # the two parities recover two different keys
    assert keys[0] != keys[1]


def test_wrong_length_rejected(signing_key):
    sig = signing_key.sign_recoverable(MESSAGE)
    digest = hashlib.sha256(MESSAGE).digest()
    for bad in (sig[:64], sig + b"\x00", b"", sig[:1]):
        with pytest.raises(MalformedSignature):
            VerifyingKey.from_recoverable_signature_with_digest(bad, digest)


def test_invalid_recovery_id_rejected(signing_key):
    sig = signing_key.sign_recoverable(MESSAGE)
    digest = hashlib.sha256(MESSAGE).digest()
    for recid in (4, 5, 127, 255):
        bad = bytes(bytearray([recid])) + sig[1:]
        with pytest.raises(MalformedSignature):
            VerifyingKey.from_recoverable_signature_with_digest(bad, digest)


def test_out_of_range_r_s_rejected(signing_key):
    sig = signing_key.sign_recoverable(MESSAGE)
    digest = hashlib.sha256(MESSAGE).digest()
    r = string_to_number(sig[1:33])
    s = string_to_number(sig[33:])
    bad_values = [
        encode_sig(0, 0, s),
        encode_sig(0, ORDER, s),
        encode_sig(0, ORDER + 1, s),
        encode_sig(0, r, 0),
        encode_sig(0, r, ORDER),
        encode_sig(0, r, ORDER + 1),
    ]
    for bad in bad_values:
        with pytest.raises(MalformedSignature):
            VerifyingKey.from_recoverable_signature_with_digest(bad, digest)


def test_recovery_id_implying_x_out_of_range_rejected(signing_key):
    digest = hashlib.sha256(MESSAGE).digest()
    # r + n >= p for r close to n, so recid with the overflow bit set
    # cannot map to a curve point
    sig = encode_sig(2, ORDER - 1, 1)
    with pytest.raises(MalformedSignature):
        VerifyingKey.from_recoverable_signature_with_digest(sig, digest)


def test_x_without_curve_point_rejected(signing_key):
    digest = hashlib.sha256(MESSAGE).digest()
    # find an x in [1, n) with no curve point
    x = 1
    while True:
        alpha = (
            pow(x, 3, P)
            + SECP256k1.curve.a() * x
            + SECP256k1.curve.b()
        ) % P
        if numbertheory.jacobi(alpha, P) == -1:
            break
        x += 1
    sig = encode_sig(0, x, 1)
    with pytest.raises(MalformedSignature):
        VerifyingKey.from_recoverable_signature_with_digest(sig, digest)


def test_other_curves_rejected():
    sk = SigningKey.generate(curve=NIST256p)
    with pytest.raises(ValueError):
        sk.sign_recoverable(MESSAGE)
    with pytest.raises(ValueError):
        sk.sign_digest_recoverable(b"\x01" * 32)

    ed_sk = SigningKey.generate(curve=Ed25519)
    with pytest.raises(ValueError):
        ed_sk.sign_recoverable(MESSAGE)

    digest = hashlib.sha256(MESSAGE).digest()
    sig = b"\x00" * 65
    with pytest.raises(ValueError):
        VerifyingKey.from_recoverable_signature(
            sig, MESSAGE, curve=NIST256p
        )
    with pytest.raises(ValueError):
        VerifyingKey.from_recoverable_signature_with_digest(
            sig, digest, curve=NIST256p
        )
    with pytest.raises(ValueError):
        VerifyingKey.from_recoverable_signature(
            sig, MESSAGE, curve=Ed25519
        )


def test_existing_signature_formats_unaffected(signing_key):
    from .util import sigencode_der, sigdecode_der

    vk = signing_key.get_verifying_key()

    raw = signing_key.sign_deterministic(MESSAGE, hashfunc=hashlib.sha256)
    assert len(raw) == 64
    assert vk.verify(raw, MESSAGE, hashfunc=hashlib.sha256)

    der = signing_key.sign_deterministic(
        MESSAGE, hashfunc=hashlib.sha256, sigencode=sigencode_der
    )
    assert vk.verify(
        der, MESSAGE, hashfunc=hashlib.sha256, sigdecode=sigdecode_der
    )

    digest = hashlib.sha256(MESSAGE).digest()
    raw_digest = signing_key.sign_digest_deterministic(
        digest, hashfunc=hashlib.sha256
    )
    assert len(raw_digest) == 64
    assert vk.verify_digest(raw_digest, digest)
