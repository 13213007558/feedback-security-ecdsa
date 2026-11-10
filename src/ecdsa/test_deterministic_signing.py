"""
Contract tests for the deterministic (RFC 6979) signing entry points.

These tests pin *what* must hold, not just that repeated calls agree:
a broken implementation can be perfectly deterministic and still derive
the wrong nonce.  All expected values below are known-answer vectors:

* ``k`` values are taken from RFC 6979 appendices A.2.3 (P-192),
  A.2.5 (P-256) and A.2.8 (P-521).  The P-192 and P-521 ``k`` values
  are additionally cross-checked against the vectors already pinned in
  ``test_pyecdsa.py`` (class ``RFC6979``).
* ``r`` is independently derivable from the RFC nonce alone:
  ``r == (k * G).x() mod n`` involves no library signing code.
* ``s`` then pins the exact digest integer ``z`` that was signed, i.e.
  it pins the truncation behaviour of ``_truncate_and_convert_digest``.

The three vectors were chosen to cover the three digest-vs-order size
relations:

* V1: digest (256 bit) LONGER than the order (192 bit) -> truncation
* V2: digest (160 bit) SHORTER than the order (256 bit) -> padding-free
* V3: order bit length (521) NOT a whole number of bytes -> the
  leftmost-bits rule of FIPS 186-4 applies to a bit-shifted digest
"""

import hashlib

import pytest

from . import rfc6979
from .ecdsa import RSZeroError
from .keys import SigningKey, BadDigestError
from .curves import NIST192p, NIST256p, NIST521p
from .util import (
    sigencode_string,
    sigencode_strings,
    sigencode_der,
    sigencode_string_canonize,
    sigencode_der_canonize,
    sigdecode_string,
    sigdecode_der,
)


class Vector(object):
    def __init__(self, name, curve, secexp, hashfunc, message, k, r, s):
        self.name = name
        self.curve = curve
        self.secexp = secexp
        self.hashfunc = hashfunc
        self.message = message
        self.digest = hashfunc(message).digest()
        self.k = k
        self.r = r
        self.s = s

    def signing_key(self):
        return SigningKey.from_secret_exponent(
            self.secexp, curve=self.curve, hashfunc=self.hashfunc
        )


# RFC 6979, appendix A.2.3 (ECDSA over P-192), key x, message "sample",
# hashed with SHA-256.  Digest (256 bit) is longer than the 192-bit
# order, so both bits2octets() and _truncate_and_convert_digest() must
# keep the *leftmost* 192 bits.  k is the RFC value; r follows from
# r = (k*G).x() mod n; s pins the truncated z.
V1 = Vector(
    "P-192/SHA-256 (digest longer than order)",
    NIST192p,
    int("6FAB034934E4C0FC9AE67F5B5659A9D7D1FEFD187EE09FD4", 16),
    hashlib.sha256,
    b"sample",
    k=int("32B1B6D7D42A05CB449065727A84804FB1A3E34D8F261496", 16),
    r=int("4B0B8CE98A92866A2820E20AA6B75B56382E0F9BFD5ECB55", 16),
    s=int("CCDB006926EA9565CBADC840829D8C384E06DE1F1E381B85", 16),
)

# RFC 6979, appendix A.2.5 (ECDSA over P-256), key x, message "sample",
# hashed with SHA-1.  Digest (160 bit) is shorter than the 256-bit
# order, so no truncation may happen anywhere.  k is the RFC value.
V2 = Vector(
    "P-256/SHA-1 (digest shorter than order)",
    NIST256p,
    int("C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721", 16),
    hashlib.sha1,
    b"sample",
    k=int("882905F1227FD620FBF2ABF21244F0BA83D0DC3A9103DBBEE43A1FB858109DB4", 16),
    r=int("61340C88C3AAEBEB4F6D667F672CA9759A6CCAA9FA8811313039EE4A35471D32", 16),
    s=int("6D7F147DAC089441BB2E2FE8F7A3FA264B9C475098FDCF6E00D7C996E1B8B7EB", 16),
)

# RFC 6979, appendix A.2.8 (ECDSA over P-521), key x, message "test",
# hashed with SHA-512.  The 521-bit order is not byte aligned, so the
# digest integer is formed from a non-byte-aligned leftmost-bits value.
# k is the RFC value (also pinned in test_pyecdsa.py).
V3 = Vector(
    "P-521/SHA-512 (order not byte aligned)",
    NIST521p,
    int(
        "0FAD06DAA62BA3B25D2FB40133DA757205DE67F5BB0018FEE8C86E1B68C7E75C"
        "AA896EB32F1F47C70855836A6D16FCC1466F6D8FBEC67DB89EC0C08B0E996B83538",
        16,
    ),
    hashlib.sha512,
    b"test",
    k=int(
        "16200813020EC986863BEDFC1B121F605C1215645018AEA1A7B215A564DE9EB1B"
        "38A67AA1128B80CE391C4FB71187654AAA3431027BFC7F395766CA988C964DC56D",
        16,
    ),
    r=int(
        "13E99020ABF5CEE7525D16B69B229652AB6BDF2AFFCAEF38773B4B7D08725F10C"
        "DB93482FDCC54EDCEE91ECA4166B2A7C6265EF0CE2BD7051B7CEF945BABD47EE6D",
        16,
    ),
    s=int(
        "1FBD0013C674AA79CB39849527916CE301C66EA7CE8B80682786AD60F98F7E78A"
        "19CA69EFF5C57400E3B3A0AD66CE0978214D13BAF4E9AC60752F7B155E2DE4DCE3",
        16,
    ),
)

VECTORS = [V1, V2, V3]
VECTOR_IDS = [v.name for v in VECTORS]


@pytest.fixture(params=VECTORS, ids=VECTOR_IDS)
def vector(request):
    return request.param


def test_nonce_matches_rfc6979(vector):
    """generate_k() must reproduce the RFC 6979 nonce exactly."""
    k = rfc6979.generate_k(
        vector.curve.order, vector.secexp, vector.hashfunc, vector.digest
    )
    assert k == vector.k


def test_signature_matches_vector(vector):
    """r and s must match the known answer, not just be self-consistent."""
    sk = vector.signing_key()
    sig = sk.sign_digest_deterministic(
        vector.digest, hashfunc=vector.hashfunc, allow_truncate=True
    )
    r, s = sigdecode_string(sig, vector.curve.order)
    assert r == vector.r
    assert s == vector.s
    # r must be the x-coordinate of the RFC nonce point: the signing
    # code may not derive the nonce any other way than generate_k()
    assert r == (vector.k * vector.curve.generator).x() % vector.curve.order


def test_vector_signature_verifies(vector):
    sk = vector.signing_key()
    vk = sk.get_verifying_key()
    sig = sk.sign_digest_deterministic(
        vector.digest, hashfunc=vector.hashfunc, allow_truncate=True
    )
    assert vk.verify_digest(
        sig,
        vector.digest,
        sigdecode=sigdecode_string,
        allow_truncate=True,
    )


def test_message_and_digest_entry_points_agree(vector):
    """sign_deterministic(data) must equal sign_digest_deterministic()
    over hashfunc(data) with allow_truncate=True -- the message-level
    entry point hardcodes allow_truncate=True (see keys.py), so the two
    are comparable only under that flag."""
    sk = vector.signing_key()
    sig_msg = sk.sign_deterministic(vector.message, hashfunc=vector.hashfunc)
    sig_dig = sk.sign_digest_deterministic(
        vector.digest, hashfunc=vector.hashfunc, allow_truncate=True
    )
    assert sig_msg == sig_dig


def test_allow_truncate_defaults_differ_between_entry_points():
    """The message-level entry point always truncates; the digest-level
    one refuses oversized digests unless told otherwise.  V1 uses a
    256-bit digest on a 192-bit curve to pin both sides of the contract."""
    sk = V1.signing_key()
    # message-level: truncation is implied, must succeed
    sk.sign_deterministic(V1.message, hashfunc=V1.hashfunc)
    # digest-level: same oversized digest is rejected by default
    with pytest.raises(BadDigestError):
        sk.sign_digest_deterministic(V1.digest, hashfunc=V1.hashfunc)


def test_allow_truncate_boundary():
    """allow_truncate=False must accept a digest of exactly baselen bytes
    and reject one byte more; with no truncation needed both flags must
    give bit-identical signatures."""
    # P-256 has baselen == 32: a SHA-256 digest sits exactly on the boundary
    sk = V2.signing_key()
    exact = hashlib.sha256(b"boundary").digest()
    assert len(exact) == V2.curve.baselen
    sig_strict = sk.sign_digest_deterministic(
        exact, hashfunc=hashlib.sha256, allow_truncate=False
    )
    sig_lax = sk.sign_digest_deterministic(
        exact, hashfunc=hashlib.sha256, allow_truncate=True
    )
    assert sig_strict == sig_lax
    with pytest.raises(BadDigestError):
        sk.sign_digest_deterministic(
            exact + b"\x00", hashfunc=hashlib.sha256, allow_truncate=False
        )


def test_truncation_keeps_leftmost_bits_p192():
    """On P-192 (baselen 24) everything past the leftmost 192 bits of the
    digest is ignored -- by the nonce derivation (bits2octets) and by the
    signed value z alike -- so appending bytes cannot change the signature."""
    sk = V1.signing_key()
    assert len(V1.digest) == 32  # already longer than baselen == 24
    sig = sk.sign_digest_deterministic(
        V1.digest, hashfunc=V1.hashfunc, allow_truncate=True
    )
    extended = sk.sign_digest_deterministic(
        V1.digest + b"\xde\xad\xbe\xef",
        hashfunc=V1.hashfunc,
        allow_truncate=True,
    )
    assert extended == sig


def test_truncation_keeps_leftmost_bits_p521():
    """P-521: qlen == 521, so of a 66-byte (528-bit) digest the low 7 bits
    are shifted out (FIPS 186-4 leftmost-bits rule).  Flipping any of the
    low 7 bits must not change the signature; flipping bit 8 must."""
    sk = V3.signing_key()
    digest66 = V3.digest + b"\x00\x00"  # 64 + 2 = 66 bytes == 528 bits
    sig = sk.sign_digest_deterministic(
        digest66, hashfunc=V3.hashfunc, allow_truncate=True
    )
    # low 7 bits of the last byte are dropped
    for low_bits in (b"\x01", b"\x2a", b"\x7f"):
        flipped = digest66[:-1] + low_bits
        assert (
            sk.sign_digest_deterministic(
                flipped, hashfunc=V3.hashfunc, allow_truncate=True
            )
            == sig
        )
    # bit 8 (0x80 of the last byte) is the lowest *retained* bit
    flipped = digest66[:-1] + b"\x80"
    assert (
        sk.sign_digest_deterministic(
            flipped, hashfunc=V3.hashfunc, allow_truncate=True
        )
        != sig
    )


def test_extra_entropy_changes_nonce_but_signature_still_verifies(vector):
    """extra_entropy feeds into the RFC 6979 HMAC seed (section 3.6), so
    it must change the nonce -- and only the nonce derivation; the
    resulting signature must still verify against the same digest."""
    order = vector.curve.order
    k_plain = rfc6979.generate_k(
        order, vector.secexp, vector.hashfunc, vector.digest
    )
    k_entropy = rfc6979.generate_k(
        order,
        vector.secexp,
        vector.hashfunc,
        vector.digest,
        extra_entropy=b"\x01" * 32,
    )
    assert k_entropy != k_plain

    sk = vector.signing_key()
    vk = sk.get_verifying_key()
    sig = sk.sign_digest_deterministic(
        vector.digest,
        hashfunc=vector.hashfunc,
        extra_entropy=b"\x01" * 32,
        allow_truncate=True,
    )
    r, s = sigdecode_string(sig, order)
    # nonce changed -> r changed -> no longer the RFC vector
    assert r != vector.r
    # entropy is mixed in deterministically
    assert sig == sk.sign_digest_deterministic(
        vector.digest,
        hashfunc=vector.hashfunc,
        extra_entropy=b"\x01" * 32,
        allow_truncate=True,
    )
    assert vk.verify_digest(
        sig, vector.digest, sigdecode=sigdecode_string, allow_truncate=True
    )


def test_retry_gen_skips_candidates(vector):
    """retry_gen must skip that many *valid* candidates: the skipped
    values are exactly what retry_gen=0 would have returned, and every
    returned k is deterministic and in range."""
    order = vector.curve.order
    k0 = rfc6979.generate_k(
        order, vector.secexp, vector.hashfunc, vector.digest, retry_gen=0
    )
    k1 = rfc6979.generate_k(
        order, vector.secexp, vector.hashfunc, vector.digest, retry_gen=1
    )
    k2 = rfc6979.generate_k(
        order, vector.secexp, vector.hashfunc, vector.digest, retry_gen=2
    )
    assert k0 == vector.k
    assert len(set([k0, k1, k2])) == 3
    for k in (k0, k1, k2):
        assert 1 <= k < order


def _generate_k_with_forced_candidates(vector, forced):
    """Run generate_k() with a controlled substitute for the candidate
    conversion: the first ``len(forced)`` Step-H3 candidates are replaced
    by the values in ``forced`` (the digest conversion in bits2octets is
    left untouched).  This triggers the zero/out-of-range rejection path
    deterministically instead of waiting for a random collision."""
    real_bits2int = rfc6979.bits2int
    calls = []

    def fake_bits2int(data, qlen):
        calls.append(1)
        # first call is bits2octets() converting the digest; the rest
        # are nonce candidates in Step H3
        candidate_no = len(calls) - 2
        if 0 <= candidate_no < len(forced):
            return forced[candidate_no]
        return real_bits2int(data, qlen)

    rfc6979.bits2int = fake_bits2int
    try:
        return rfc6979.generate_k(
            vector.curve.order,
            vector.secexp,
            vector.hashfunc,
            vector.digest,
        )
    finally:
        rfc6979.bits2int = real_bits2int


def test_rejected_candidates_are_skipped(vector):
    """A candidate of 0 or >= order must be rejected (RFC 6979 step H3)
    and the HMAC stream advanced exactly like retry_gen does: forcing N
    bad candidates must yield the same k as retry_gen=N."""
    order = vector.curve.order
    expected_1 = rfc6979.generate_k(
        order, vector.secexp, vector.hashfunc, vector.digest, retry_gen=1
    )
    expected_2 = rfc6979.generate_k(
        order, vector.secexp, vector.hashfunc, vector.digest, retry_gen=2
    )
    # zero candidate
    assert _generate_k_with_forced_candidates(vector, [0]) == expected_1
    # out-of-range candidate (== order)
    assert _generate_k_with_forced_candidates(vector, [order]) == expected_1
    # two rejections in a row
    assert _generate_k_with_forced_candidates(vector, [0, order]) == expected_2


def test_sign_digest_deterministic_retries_on_rszero():
    """When sign_digest() raises RSZeroError (r == 0 or s == 0), the
    deterministic entry point must retry with the *next* RFC 6979
    candidate (retry_gen=1), not re-use or re-seed the nonce.  The
    failure is injected by a controlled substitute for sign_digest."""
    sk = V1.signing_key()
    order = V1.curve.order
    k0 = rfc6979.generate_k(order, V1.secexp, V1.hashfunc, V1.digest)
    k1 = rfc6979.generate_k(
        order, V1.secexp, V1.hashfunc, V1.digest, retry_gen=1
    )

    real_sign_digest = sk.sign_digest
    seen_k = []

    def flaky_sign_digest(digest, sigencode=None, k=None, allow_truncate=False):
        seen_k.append(k)
        if len(seen_k) == 1:
            raise RSZeroError("controlled substitute: unlucky r or s")
        return real_sign_digest(
            digest, sigencode=sigencode, k=k, allow_truncate=allow_truncate
        )

    sk.sign_digest = flaky_sign_digest
    try:
        sig = sk.sign_digest_deterministic(
            V1.digest, hashfunc=V1.hashfunc, allow_truncate=True
        )
    finally:
        sk.sign_digest = real_sign_digest

    # first attempt used the RFC nonce, the retry used the next candidate
    assert seen_k == [k0, k1]
    # and the result equals signing directly with that second candidate
    assert sig == real_sign_digest(
        V1.digest, sigencode=sigencode_string, k=k1, allow_truncate=True
    )
    assert V1.signing_key().get_verifying_key().verify_digest(
        sig, V1.digest, sigdecode=sigdecode_string, allow_truncate=True
    )


def test_encoding_does_not_affect_nonce(vector):
    """DER, raw-string and low-S (canonized) encodings only change the
    representation (or pick the equivalent signature (r, n-s)): the
    nonce -- and therefore r -- must be identical for all of them."""
    sk = vector.signing_key()
    order = vector.curve.order

    raw = sk.sign_digest_deterministic(
        vector.digest, hashfunc=vector.hashfunc, allow_truncate=True
    )
    der = sk.sign_digest_deterministic(
        vector.digest,
        hashfunc=vector.hashfunc,
        sigencode=sigencode_der,
        allow_truncate=True,
    )
    low_s = sk.sign_digest_deterministic(
        vector.digest,
        hashfunc=vector.hashfunc,
        sigencode=sigencode_string_canonize,
        allow_truncate=True,
    )
    low_s_der = sk.sign_digest_deterministic(
        vector.digest,
        hashfunc=vector.hashfunc,
        sigencode=sigencode_der_canonize,
        allow_truncate=True,
    )

    r_raw, s_raw = sigdecode_string(raw, order)
    r_der, s_der = sigdecode_der(der, order)
    r_low, s_low = sigdecode_string(low_s, order)
    r_low_der, s_low_der = sigdecode_der(low_s_der, order)

    # same nonce everywhere: r is fixed by k alone
    assert (r_raw, r_der, r_low, r_low_der) == (vector.r,) * 4
    # plain encodings carry the RFC s
    assert s_raw == vector.s
    assert s_der == vector.s
    # canonized encodings carry the equivalent low-S representative
    assert s_low == s_low_der == min(vector.s, order - vector.s)

    # every representation verifies against the same digest
    vk = sk.get_verifying_key()
    assert vk.verify_digest(
        raw, vector.digest, sigdecode=sigdecode_string, allow_truncate=True
    )
    assert vk.verify_digest(
        der, vector.digest, sigdecode=sigdecode_der, allow_truncate=True
    )
    # the low-S form is an equivalent signature: it verifies even with
    # the plain decoder
    assert vk.verify_digest(
        low_s, vector.digest, sigdecode=sigdecode_string, allow_truncate=True
    )
