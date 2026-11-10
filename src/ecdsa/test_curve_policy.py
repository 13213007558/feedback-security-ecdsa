"""Tests for the optional curve admission policy (CurvePolicy).

The matrix tests feed the same four logical inputs through every public
import path and assert that the conclusion (accept/reject and the stage at
which the rejection happens) is consistent:

1. a trusted *named* curve,
2. *equivalent explicit* parameters of a trusted curve,
3. explicit parameters with *a single parameter changed*,
4. a *different curve with the same field size* (NIST256p vs SECP256k1).
"""

import pytest

from . import der
from ._compat import bit_length
from .curves import Curve, NIST256p, SECP256k1, Ed25519, UnknownCurveError
from .curve_policy import (
    CurvePolicy,
    CurveAdmissionError,
    curves_equal,
    DEFAULT_MAX_BITS,
)
from .ecdh import ECDH, InvalidCurveError
from .errors import MalformedPointError
from .keys import SigningKey, VerifyingKey
from .util import number_to_string


# -- helpers to build the four classes of input -----------------------------


def encode_explicit_parameters(curve, **overrides):
    """Build a RFC5912 ECParameters explicit-parameters DER blob."""
    prime = overrides.get("prime", curve.curve.p())
    coeff_a = overrides.get("a", curve.curve.a())
    coeff_b = overrides.get("b", curve.curve.b())
    order = overrides.get("order", curve.order)
    cofactor = overrides.get("cofactor", curve.curve.cofactor())
    base_x = overrides.get("base_x", curve.generator.x())
    base_y = overrides.get("base_y", curve.generator.y())

    base_point = (
        b"\x04"
        + number_to_string(base_x, prime)
        + number_to_string(base_y, prime)
    )
    field_id = der.encode_sequence(
        der.encode_oid(1, 2, 840, 10045, 1, 1), der.encode_integer(prime)
    )
    curve_eq = der.encode_sequence(
        der.encode_octet_string(number_to_string(coeff_a % prime, prime)),
        der.encode_octet_string(number_to_string(coeff_b % prime, prime)),
    )
    elements = [
        der.encode_integer(1),
        field_id,
        curve_eq,
        der.encode_octet_string(base_point),
        der.encode_integer(order),
    ]
    if cofactor is not None:
        elements.append(der.encode_integer(cofactor))
    return der.encode_sequence(*elements)


def wrap_spki(curve_parameters, point_bytes):
    """Wrap curve parameters and a point into a SubjectPublicKeyInfo DER."""
    from .util import encoded_oid_ecPublicKey

    return der.encode_sequence(
        der.encode_sequence(encoded_oid_ecPublicKey, curve_parameters),
        der.encode_bitstring(point_bytes, 0),
    )


def wrap_ssleay_privkey(curve_parameters, privkey_bytes, point_bytes):
    """Wrap curve parameters into an RFC5915 EC private key DER."""
    return der.encode_sequence(
        der.encode_integer(1),
        der.encode_octet_string(privkey_bytes),
        der.encode_constructed(0, curve_parameters),
        der.encode_constructed(
            1, der.encode_bitstring(point_bytes, 0)
        ),
    )


def wrap_pkcs8_privkey(curve_parameters, privkey_bytes):
    """Wrap curve parameters into an RFC5958 PKCS#8 private key DER."""
    from .util import oid_ecPublicKey

    inner = der.encode_sequence(
        der.encode_integer(1),
        der.encode_octet_string(privkey_bytes),
    )
    return der.encode_sequence(
        der.encode_integer(0),
        der.encode_sequence(
            der.encode_oid(*oid_ecPublicKey), curve_parameters
        ),
        der.encode_octet_string(inner),
    )


@pytest.fixture
def trusted_key():
    return SigningKey.generate(curve=NIST256p)


@pytest.fixture
def policy():
    return CurvePolicy(allowed_curves=[NIST256p], allow_explicit=True)


@pytest.fixture
def named_only_policy():
    return CurvePolicy(allowed_curves=[NIST256p])


# -- policy object itself ----------------------------------------------------


class TestCurvePolicyBasics:
    def test_default_policy_knows_all_library_curves(self):
        pol = CurvePolicy()
        from .curves import curves

        assert sorted(c.name for c in pol.allowed_curves) == sorted(
            c.name for c in curves
        )
        assert pol.allow_explicit is False
        assert pol.max_field_bits == DEFAULT_MAX_BITS

    def test_empty_allowed_curves_rejected(self):
        with pytest.raises(ValueError):
            CurvePolicy(allowed_curves=[])

    def test_invalid_limits_rejected(self):
        with pytest.raises(ValueError):
            CurvePolicy(max_field_bits=0)
        with pytest.raises(ValueError):
            CurvePolicy(max_order_bits=-1)

    def test_named_curve_admitted(self):
        pol = CurvePolicy(allowed_curves=[NIST256p])
        assert pol.admit(NIST256p, explicit=False) is NIST256p

    def test_named_curve_not_in_policy_rejected(self):
        pol = CurvePolicy(allowed_curves=[NIST256p])
        with pytest.raises(CurveAdmissionError):
            pol.admit(SECP256k1, explicit=False)

    def test_explicit_disabled_rejects_even_equivalent_params(self):
        pol = CurvePolicy(allowed_curves=[NIST256p], allow_explicit=False)
        with pytest.raises(CurveAdmissionError):
            pol.admit(NIST256p, explicit=True)

    def test_full_parameter_comparison_takes_order_and_cofactor(self):
        # the plain Curve equality ignores the order and cofactor; the policy
        # helper must not. Build the "tmp_curve" objects directly, the same
        # way Curve.from_der does before asking the policy to identify them.
        from .ellipticcurve import CurveFp, PointJacobi

        def tmp_curve(prime, a, b, order, cofactor):
            curve_fp = CurveFp(prime, a, b, cofactor)
            base = PointJacobi.from_bytes(
                curve_fp,
                NIST256p.generator.to_bytes("uncompressed"),
                valid_encodings=(
                    "uncompressed",
                    "compressed",
                    "hybrid",
                ),
                order=order,
                generator=True,
            )
            return Curve("unknown", curve_fp, base, None)

        equivalent = tmp_curve(
            NIST256p.curve.p(),
            NIST256p.curve.a(),
            NIST256p.curve.b(),
            NIST256p.order,
            NIST256p.curve.cofactor(),
        )
        assert curves_equal(NIST256p, equivalent)

        changed_order = tmp_curve(
            NIST256p.curve.p(),
            NIST256p.curve.a(),
            NIST256p.curve.b(),
            NIST256p.order + 2,
            NIST256p.curve.cofactor(),
        )
        assert changed_order == NIST256p  # Curve.__eq__ misses the order
        assert not curves_equal(NIST256p, changed_order)

        changed_h = tmp_curve(
            NIST256p.curve.p(),
            NIST256p.curve.a(),
            NIST256p.curve.b(),
            NIST256p.order,
            NIST256p.curve.cofactor() + 1,
        )
        assert changed_h == NIST256p  # Curve.__eq__ misses the cofactor
        assert not curves_equal(NIST256p, changed_h)


# -- Curve.from_der / from_pem stage -----------------------------------------


class TestCurveFromDerPolicy:
    def test_named_curve_admitted(self, policy):
        assert Curve.from_der(NIST256p.to_der("named_curve"),
                              curve_policy=policy) is NIST256p

    def test_named_curve_oid_not_allowed(self):
        pol = CurvePolicy(allowed_curves=[SECP256k1])
        with pytest.raises(CurveAdmissionError):
            Curve.from_der(NIST256p.to_der("named_curve"), curve_policy=pol)

    def test_equivalent_explicit_admitted_and_resolved(self, policy):
        decoded = Curve.from_der(
            encode_explicit_parameters(NIST256p), curve_policy=policy
        )
        assert decoded is NIST256p

    def test_explicit_rejected_when_disabled(self, named_only_policy):
        with pytest.raises(CurveAdmissionError):
            Curve.from_der(
                encode_explicit_parameters(NIST256p),
                curve_policy=named_only_policy,
            )

    @pytest.mark.parametrize(
        "name,overrides",
        [
            ("coefficient_b", dict(b=lambda c: c.curve.b() + 1)),
            ("coefficient_a", dict(a=lambda c: c.curve.a() + 1)),
            ("order", dict(order=lambda c: c.order + 2)),
            ("cofactor", dict(cofactor=lambda c: c.curve.cofactor() + 1)),
            ("base_point", dict(base_y=lambda c: c.generator.y() + 1)),
        ],
    )
    def test_single_parameter_change_rejected(self, policy, name, overrides):
        concrete = dict(
            (key, value(NIST256p)) for key, value in overrides.items()
        )
        blob = encode_explicit_parameters(NIST256p, **concrete)
        with pytest.raises(CurveAdmissionError):
            Curve.from_der(blob, curve_policy=policy)

    def test_same_size_other_curve_rejected(self, policy):
        # SECP256k1 also has a 256-bit prime, but different domain parameters
        assert bit_length(NIST256p.curve.p()) == bit_length(
            SECP256k1.curve.p()
        )
        with pytest.raises(CurveAdmissionError):
            Curve.from_der(
                encode_explicit_parameters(SECP256k1), curve_policy=policy
            )

    def test_oversized_prime_rejected_before_point_decoding(
        self, named_only_policy
    ):
        pol = CurvePolicy(
            allowed_curves=[NIST256p],
            allow_explicit=True,
            max_field_bits=256,
            max_order_bits=256,
        )
        huge_prime = 1 << 2048
        blob = encode_explicit_parameters(
            NIST256p,
            prime=huge_prime,
            a=0,
            b=1,
            order=huge_prime - 1,
            base_x=1,
            base_y=1,
            cofactor=1,
        )
        # rejected by the structural check, not by (expensive) point math
        with pytest.raises(CurveAdmissionError) as exc:
            Curve.from_der(blob, curve_policy=pol)
        assert "field prime" in str(exc.value)

    def test_oversized_order_rejected_structurally(self):
        pol = CurvePolicy(
            allowed_curves=[NIST256p], allow_explicit=True, max_order_bits=256
        )
        blob = encode_explicit_parameters(
            NIST256p, order=1 << 2048
        )
        with pytest.raises(CurveAdmissionError) as exc:
            Curve.from_der(blob, curve_policy=pol)
        assert "base point order" in str(exc.value)

    def test_invalid_prime_rejected_structurally(self, policy):
        # build the blob manually: a two-bit prime can't encode a real base
        # point, but the structural check must reject it before point math
        base_point = b"\x04\x01\x01"
        blob = der.encode_sequence(
            der.encode_integer(1),
            der.encode_sequence(
                der.encode_oid(1, 2, 840, 10045, 1, 1), der.encode_integer(2)
            ),
            der.encode_sequence(
                der.encode_octet_string(b"\x00"),
                der.encode_octet_string(b"\x01"),
            ),
            der.encode_octet_string(base_point),
            der.encode_integer(3),
        )
        with pytest.raises(CurveAdmissionError) as exc:
            Curve.from_der(blob, curve_policy=policy)
        assert "field prime" in str(exc.value)

    def test_negative_cofactor_rejected_structurally(self, policy):
        blob = encode_explicit_parameters(NIST256p, cofactor=0)
        with pytest.raises(CurveAdmissionError):
            Curve.from_der(blob, curve_policy=policy)

    def test_from_pem_equivalent_path(self, policy):
        pem = NIST256p.to_pem("explicit")
        assert Curve.from_pem(pem, curve_policy=policy) is NIST256p

    def test_default_behaviour_without_policy_unchanged(self):
        # an unknown custom curve remains acceptable without a policy
        from .ellipticcurve import CurveFp, PointJacobi

        curve_fp = CurveFp(23, 1, 7)
        base = PointJacobi(curve_fp, 13, 3, 1, 9, generator=True)
        custom = Curve("unknown", curve_fp, base, None)
        assert Curve.from_der(custom.to_der()) == custom


# -- consistent matrix over the public key import paths ----------------------


class TestImportMatrixConsistency:
    def _materials(self, key, parameters_der):
        vk = key.get_verifying_key()
        point = vk.to_string("uncompressed")
        spki = wrap_spki(parameters_der, point)
        pem = der.topem(spki, "PUBLIC KEY")
        priv = key.to_string()
        return vk, point, spki, pem, priv

    def test_named_trusted_curve_accepted_everywhere(self, policy):
        key = SigningKey.generate(curve=NIST256p)
        vk, point, spki, pem, priv = self._materials(
            key, NIST256p.to_der("named_curve")
        )
        VerifyingKey.from_der(spki, curve_policy=policy)
        VerifyingKey.from_pem(pem, curve_policy=policy)
        VerifyingKey.from_string(point, NIST256p, curve_policy=policy)
        VerifyingKey.from_public_point(
            vk.pubkey.point, NIST256p, curve_policy=policy
        )
        ssleay = wrap_ssleay_privkey(
            NIST256p.to_der("named_curve"), priv, point
        )
        pkcs8 = wrap_pkcs8_privkey(NIST256p.to_der("named_curve"), priv)
        SigningKey.from_der(ssleay, curve_policy=policy)
        SigningKey.from_der(pkcs8, curve_policy=policy)
        SigningKey.from_pem(der.topem(ssleay, "EC PRIVATE KEY"),
                            curve_policy=policy)
        SigningKey.from_pem(der.topem(pkcs8, "PRIVATE KEY"),
                            curve_policy=policy)
        SigningKey.from_string(priv, NIST256p, curve_policy=policy)

    def test_equivalent_explicit_accepted_everywhere(self, policy):
        key = SigningKey.generate(curve=NIST256p)
        params = encode_explicit_parameters(NIST256p)
        vk, point, spki, pem, priv = self._materials(key, params)
        assert VerifyingKey.from_der(spki,
                                     curve_policy=policy).curve is NIST256p
        assert VerifyingKey.from_pem(pem,
                                     curve_policy=policy).curve is NIST256p
        ssleay = wrap_ssleay_privkey(params, priv, point)
        pkcs8 = wrap_pkcs8_privkey(params, priv)
        assert SigningKey.from_der(ssleay,
                                   curve_policy=policy).curve is NIST256p
        assert SigningKey.from_der(pkcs8,
                                   curve_policy=policy).curve is NIST256p

    @pytest.mark.parametrize(
        "name,overrides",
        [
            ("coefficient_b", dict(b=lambda c: c.curve.b() + 1)),
            ("coefficient_a", dict(a=lambda c: c.curve.a() + 1)),
            ("order", dict(order=lambda c: c.order + 2)),
            ("cofactor", dict(cofactor=lambda c: c.curve.cofactor() + 1)),
            ("base_point", dict(base_y=lambda c: c.generator.y() + 1)),
        ],
    )
    def test_tampered_explicit_rejected_everywhere(self, policy, name,
                                                   overrides):
        key = SigningKey.generate(curve=NIST256p)
        concrete = dict(
            (key_name, value(NIST256p))
            for key_name, value in overrides.items()
        )
        params = encode_explicit_parameters(NIST256p, **concrete)
        vk, point, spki, pem, priv = self._materials(key, params)

        # parsing may already fail for some tampering, otherwise the policy
        # must reject at the parameter-identification stage; in no case may
        # a key object be returned
        for call in (
            lambda: Curve.from_der(params, curve_policy=policy),
            lambda: VerifyingKey.from_der(spki, curve_policy=policy),
            lambda: VerifyingKey.from_pem(pem, curve_policy=policy),
        ):
            with pytest.raises((CurveAdmissionError, MalformedPointError,
                                der.UnexpectedDER, UnknownCurveError,
                                AssertionError)):
                call()

        ssleay = wrap_ssleay_privkey(params, priv, point)
        pkcs8 = wrap_pkcs8_privkey(params, priv)
        for call in (
            lambda: SigningKey.from_der(ssleay, curve_policy=policy),
            lambda: SigningKey.from_der(pkcs8, curve_policy=policy),
        ):
            with pytest.raises((CurveAdmissionError, MalformedPointError,
                                der.UnexpectedDER, UnknownCurveError,
                                AssertionError)):
                call()

    def test_same_size_other_curve_rejected_everywhere(self, policy):
        # class four: SECP256k1 has an equally wide field but is not trusted
        key = SigningKey.generate(curve=SECP256k1)
        vk = key.get_verifying_key()
        point = vk.to_string("uncompressed")
        spki_named = wrap_spki(SECP256k1.to_der("named_curve"), point)
        spki_explicit = wrap_spki(
            encode_explicit_parameters(SECP256k1), point
        )
        for blob in (spki_named, spki_explicit):
            with pytest.raises(CurveAdmissionError):
                VerifyingKey.from_der(blob, curve_policy=policy)
            with pytest.raises(CurveAdmissionError):
                VerifyingKey.from_pem(
                    der.topem(blob, "PUBLIC KEY"), curve_policy=policy
                )
        with pytest.raises(CurveAdmissionError):
            VerifyingKey.from_string(point, SECP256k1, curve_policy=policy)
        with pytest.raises(CurveAdmissionError):
            SigningKey.from_string(
                key.to_string(), SECP256k1, curve_policy=policy
            )

    def test_eddsa_named_curve_policy(self):
        key = SigningKey.generate(curve=Ed25519)
        pem = key.get_verifying_key().to_pem()
        priv_pem = key.to_pem(format="pkcs8")

        allow = CurvePolicy(allowed_curves=[Ed25519])
        assert VerifyingKey.from_pem(pem,
                                     curve_policy=allow).curve is Ed25519
        assert SigningKey.from_pem(priv_pem,
                                   curve_policy=allow).curve is Ed25519

        deny = CurvePolicy(allowed_curves=[NIST256p])
        with pytest.raises(CurveAdmissionError):
            VerifyingKey.from_pem(pem, curve_policy=deny)
        with pytest.raises(CurveAdmissionError):
            SigningKey.from_pem(priv_pem, curve_policy=deny)


# -- independence from the existing checks -----------------------------------


class TestExistingChecksUnaffected:
    def test_off_curve_point_still_rejected(self, policy):
        key = SigningKey.generate(curve=NIST256p)
        raw = bytearray(key.get_verifying_key().to_string())
        raw[-1] ^= 1
        with pytest.raises(MalformedPointError):
            VerifyingKey.from_string(bytes(raw), NIST256p,
                                     curve_policy=policy)

    def test_valid_encodings_still_enforced(self, policy):
        key = SigningKey.generate(curve=NIST256p)
        spki = key.get_verifying_key().to_der()
        with pytest.raises(MalformedPointError):
            VerifyingKey.from_der(
                spki,
                curve_policy=policy,
                valid_encodings=set(["compressed"]),
            )

    def test_valid_curve_encodings_still_enforced(self, named_only_policy):
        key = SigningKey.generate(curve=NIST256p)
        explicit_spki = key.get_verifying_key().to_der(
            curve_parameters_encoding="explicit"
        )
        with pytest.raises(der.UnexpectedDER):
            VerifyingKey.from_der(
                explicit_spki,
                curve_policy=named_only_policy,
                valid_curve_encodings=["named_curve"],
            )

    def test_no_policy_keeps_default_acceptance(self):
        # default call must stay compatible: no policy argument at all
        key = SigningKey.generate(curve=NIST256p)
        vk = VerifyingKey.from_der(key.get_verifying_key().to_der())
        assert vk.curve is NIST256p
        SigningKey.from_der(key.to_der())


# -- ECDH --------------------------------------------------------------------


class TestECDHPolicy:
    def test_happy_path_named_and_explicit(self):
        pol = CurvePolicy(allowed_curves=[NIST256p], allow_explicit=True)
        local = SigningKey.generate(curve=NIST256p)
        peer = SigningKey.generate(curve=NIST256p)

        ecdh = ECDH(curve=NIST256p, curve_policy=pol)
        ecdh.load_private_key(local)
        ecdh.load_received_public_key_der(
            peer.get_verifying_key().to_der(
                curve_parameters_encoding="explicit"
            )
        )
        secret = ecdh.generate_sharedsecret_bytes()
        assert len(secret) == NIST256p.verifying_key_length // 2

    def test_peer_disallowed_curve_rejected_at_import(self):
        pol = CurvePolicy(allowed_curves=[NIST256p])
        ecdh = ECDH(curve=NIST256p, curve_policy=pol)
        ecdh.generate_private_key()
        bad = SigningKey.generate(curve=SECP256k1).get_verifying_key()
        with pytest.raises(CurveAdmissionError):
            ecdh.load_received_public_key(bad)

    def test_local_disallowed_curve_rejected_at_import(self):
        pol = CurvePolicy(allowed_curves=[NIST256p])
        ecdh = ECDH(curve_policy=pol)
        with pytest.raises(CurveAdmissionError):
            ecdh.load_private_key(SigningKey.generate(curve=SECP256k1))

    def test_set_curve_enforces_policy(self):
        pol = CurvePolicy(allowed_curves=[NIST256p])
        ecdh = ECDH(curve_policy=pol)
        with pytest.raises(CurveAdmissionError):
            ecdh.set_curve(SECP256k1)

    def test_injected_bad_key_rejected_at_use_stage(self):
        # import can't be bypassed by replacing the attribute afterwards;
        # the shared-secret operation re-checks both sides
        pol = CurvePolicy(allowed_curves=[NIST256p])
        ecdh = ECDH(curve=NIST256p, curve_policy=pol)
        ecdh.generate_private_key()
        ecdh.public_key = SigningKey.generate(
            curve=SECP256k1
        ).get_verifying_key()
        with pytest.raises(CurveAdmissionError):
            ecdh.generate_sharedsecret_bytes()

    def test_curve_swap_after_import_still_rejected(self):
        # even if admission were dodged, the curve mismatch check remains
        pol = CurvePolicy(allowed_curves=[NIST256p], allow_explicit=True)
        ecdh = ECDH(curve=NIST256p, curve_policy=pol)
        ecdh.generate_private_key()
        ecdh.public_key = SigningKey.generate(
            curve=SECP256k1
        ).get_verifying_key()
        with pytest.raises((CurveAdmissionError, InvalidCurveError)):
            ecdh.generate_sharedsecret_bytes()

    def test_der_and_pem_wrappers_enforce_policy(self):
        pol = CurvePolicy(allowed_curves=[NIST256p])
        local = SigningKey.generate(curve=NIST256p)
        peer = SigningKey.generate(curve=NIST256p)

        ecdh = ECDH(curve=NIST256p, curve_policy=pol)
        ecdh.load_private_key_der(local.to_der())
        ecdh.load_private_key_pem(local.to_pem())
        ecdh.load_received_public_key_der(peer.get_verifying_key().to_der())
        ecdh.load_received_public_key_pem(
            peer.get_verifying_key().to_pem()
        )
        ecdh.generate_sharedsecret_bytes()

    def test_bytes_wrappers_enforce_policy(self):
        pol = CurvePolicy(allowed_curves=[NIST256p])
        local = SigningKey.generate(curve=NIST256p)
        peer = SigningKey.generate(curve=NIST256p)

        ecdh = ECDH(curve=NIST256p, curve_policy=pol)
        ecdh.load_private_key_bytes(local.to_string())
        ecdh.load_received_public_key_bytes(
            peer.get_verifying_key().to_string()
        )
        ecdh.generate_sharedsecret_bytes()

    def test_explicit_peer_rejected_when_policy_is_named_only(self):
        pol = CurvePolicy(allowed_curves=[NIST256p])
        ecdh = ECDH(curve=NIST256p, curve_policy=pol)
        ecdh.generate_private_key()
        explicit_spki = SigningKey.generate(
            curve=NIST256p
        ).get_verifying_key().to_der(curve_parameters_encoding="explicit")
        with pytest.raises(CurveAdmissionError):
            ecdh.load_received_public_key_der(explicit_spki)

    def test_default_ecdh_without_policy_unchanged(self):
        local = SigningKey.generate(curve=NIST256p)
        peer = SigningKey.generate(curve=NIST256p)
        ecdh = ECDH(curve=NIST256p)
        ecdh.load_private_key(local)
        ecdh.load_received_public_key(peer.get_verifying_key())
        ecdh.generate_sharedsecret_bytes()
