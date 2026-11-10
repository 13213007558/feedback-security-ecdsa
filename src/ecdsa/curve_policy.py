"""
Optional admission policy for elliptic curves imported from untrusted input.

The policy object describes what a verification service is willing to accept:

* the exact set of named curves that are trusted,
* whether the explicit (RFC 5912 ECParameters) encoding is accepted at all,
* the maximum size of the field and the base point order, so that an attacker
  can't make us allocate and operate on arbitrarily large parameters.

When explicit parameters are allowed, they are accepted only if *all* the
domain parameters identify a trusted curve: the field prime, the curve
equation coefficients, the base point, the order of the base point and the
cofactor are compared. Matching only the bit size, the curve name or just the
base point is not enough.
"""

from ._compat import bit_length


__all__ = ["CurvePolicy", "CurveAdmissionError"]


#: Default upper bound for the field prime and the base point order.
#: It's comfortably above the largest curve shipped with the library
#: (NIST521p / Ed448) while still being cheap to process.
DEFAULT_MAX_BITS = 1024


class CurveAdmissionError(Exception):
    """Raised when a curve is rejected by the admission policy.

    The message always states *why* the curve was rejected, so callers can
    distinguish a structural/size problem, a disallowed encoding, an unknown
    curve and parameters that merely look similar to a trusted curve.
    """

    pass


def curves_equal(trusted, candidate):
    """Compare the full set of domain parameters of two curves.

    Unlike :py:meth:`ecdsa.curves.Curve.__eq__`, this comparison also takes
    the order of the base point and the cofactor into account, so a curve
    that shares only the field (bit size) or only the equation with a trusted
    one is not treated as equivalent.

    :param trusted: a curve from the trusted, built-in set
    :type trusted: ecdsa.curves.Curve
    :param candidate: a curve decoded from untrusted input
    :type candidate: ecdsa.curves.Curve

    :rtype: bool
    """
    if trusted.curve != candidate.curve:
        # field prime and curve equation coefficients (a, b or a, d)
        return False
    if trusted.generator != candidate.generator:
        # base point coordinates
        return False
    if trusted.order != candidate.order:
        # order of the base point
        return False
    if trusted.curve.cofactor() != candidate.curve.cofactor():
        # cofactor of the curve
        return False
    return True


class CurvePolicy(object):
    """Admission policy controlling which curves may be imported.

    Construct one of these and pass it as the ``curve_policy`` argument to the
    public import methods (:py:meth:`~ecdsa.keys.VerifyingKey.from_der`,
    :py:meth:`~ecdsa.keys.VerifyingKey.from_pem`,
    :py:meth:`~ecdsa.keys.VerifyingKey.from_string`,
    :py:meth:`~ecdsa.keys.VerifyingKey.from_public_point` and the matching
    :py:class:`~ecdsa.keys.SigningKey` and :py:class:`~ecdsa.ecdh.ECDH`
    methods). When no policy is provided, every method keeps its historical,
    unrestricted behaviour.

    :param allowed_curves: iterable of trusted
        :py:class:`~ecdsa.curves.Curve` objects. Only curves that are in this
        set (compared by full domain parameters, not by name) are accepted.
        Defaults to every curve shipped with the library.
    :param bool allow_explicit: whether the explicit ECParameters encoding is
        accepted. When ``False`` only the named-curve (OID) encoding is
        allowed. When ``True`` explicit parameters are accepted only when they
        are fully equivalent to one of the ``allowed_curves``.
    :param int max_field_bits: maximum accepted bit length of the field prime
        of explicitly provided parameters; checked before any expensive
        operation is performed on the parameters.
    :param int max_order_bits: maximum accepted bit length of the base point
        order of explicitly provided parameters; checked before any expensive
        operation is performed on the parameters.
    """

    def __init__(
        self,
        allowed_curves=None,
        allow_explicit=False,
        max_field_bits=DEFAULT_MAX_BITS,
        max_order_bits=DEFAULT_MAX_BITS,
    ):
        if allowed_curves is None:
            from .curves import curves as _all_curves

            allowed = list(_all_curves)
        else:
            allowed = list(allowed_curves)
        if not allowed:
            raise ValueError("CurvePolicy needs at least one allowed curve")
        if max_field_bits <= 0 or max_order_bits <= 0:
            raise ValueError("Curve size limits must be positive integers")
        self._allowed_curves = allowed
        self.allow_explicit = bool(allow_explicit)
        self.max_field_bits = int(max_field_bits)
        self.max_order_bits = int(max_order_bits)

    @property
    def allowed_curves(self):
        """Tuple of the curves the policy accepts."""
        return tuple(self._allowed_curves)

    def check_structure(self, prime, order, cofactor, base_point_bytes):
        """Apply cheap structural and size limits to explicit parameters.

        This runs *before* the parameters are turned into a curve object and
        before the base point is decoded, so an attacker can't make the
        library perform expensive modular arithmetic on oversized values.

        :param int prime: field prime from the ECParameters structure
        :param int order: base point order from the ECParameters structure
        :param cofactor: cofactor from the ECParameters structure (may be
            ``None`` when it was omitted)
        :param bytes base_point_bytes: raw encoded base point
        """
        if prime <= 2:
            raise CurveAdmissionError(
                "Rejected explicit curve parameters: field prime is not a "
                "valid prime"
            )
        if order <= 1:
            raise CurveAdmissionError(
                "Rejected explicit curve parameters: base point order is not "
                "a positive integer greater than one"
            )
        if bit_length(prime) > self.max_field_bits:
            raise CurveAdmissionError(
                "Rejected explicit curve parameters: field prime of {0} "
                "bits exceeds the {1} bit limit".format(
                    bit_length(prime), self.max_field_bits
                )
            )
        if bit_length(order) > self.max_order_bits:
            raise CurveAdmissionError(
                "Rejected explicit curve parameters: base point order of {0} "
                "bits exceeds the {1} bit limit".format(
                    bit_length(order), self.max_order_bits
                )
            )
        if cofactor is not None and cofactor <= 0:
            raise CurveAdmissionError(
                "Rejected explicit curve parameters: cofactor must be a "
                "positive integer"
            )
        if not base_point_bytes:
            raise CurveAdmissionError(
                "Rejected explicit curve parameters: empty base point "
                "encoding"
            )

    def admit(self, curve, explicit):
        """Check whether a decoded curve is admitted by the policy.

        :param curve: the curve that was decoded (named or built from explicit
            parameters)
        :type curve: ecdsa.curves.Curve
        :param bool explicit: ``True`` if the curve was reconstructed from
            explicit ECParameters, ``False`` if it was referenced by OID

        :raises CurveAdmissionError: if the curve is not admitted

        :return: the matching trusted curve object, so callers always end up
            working with the built-in instance
        :rtype: ecdsa.curves.Curve
        """
        if explicit and not self.allow_explicit:
            raise CurveAdmissionError(
                "Rejected curve {0!r}: explicit curve parameters are not "
                "allowed by the policy".format(curve.name)
            )
        for trusted in self._allowed_curves:
            if curves_equal(trusted, curve):
                return trusted
        raise CurveAdmissionError(
            "Rejected curve {0!r}: it is not among the curves allowed by the "
            "policy (or its explicit domain parameters do not fully match a "
            "trusted curve)".format(curve.name)
        )

    def admit_curve(self, curve):
        """Admit a curve object that was obtained outside DER parsing.

        The curve may be a built-in named curve (for example one passed to
        :py:meth:`~ecdsa.keys.VerifyingKey.from_string`) or an ``"unknown"``
        curve reconstructed earlier from explicit parameters; both are checked
        against the same trusted set with the same full-parameter comparison.

        :raises CurveAdmissionError: if the curve is not admitted

        :rtype: ecdsa.curves.Curve
        """
        # A curve with no OID was built from explicit parameters.
        return self.admit(curve, explicit=curve.oid is None)
