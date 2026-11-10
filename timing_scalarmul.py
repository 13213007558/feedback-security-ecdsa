"""
Reproducible timing illustration for python-ecdsa side-channel review.

It does NOT prove (or disprove) constant-time behavior; finite-sample timing
on a general-purpose OS cannot do that. It only shows that the scalar
multiplication used on the private signing path has data-dependent,
branch-driven timing structure visible at coarse resolution.

Run: python timing_scalarmul.py [iterations]
"""
import os
import sys
import time
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from ecdsa import NIST256p, SigningKey  # noqa: E402
from ecdsa.ellipticcurve import AbstractPoint  # noqa: E402

G = NIST256p.generator
n = G.order()
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def bench(scalar, iters):
    # warm up precompute table (generator points precompute deterministically)
    G._maybe_precompute()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _ = G * scalar
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    return samples


def summ(name, samples):
    med = statistics.median(samples)
    p95 = sorted(samples)[int(len(samples) * 0.95)]
    print("%-28s median=%8.1fus  p95=%8.1fus  stdev=%8.1fus"
          % (name, med / 1000, p95 / 1000, statistics.stdev(samples) / 1000))


# k1: sparse scalar (single high bit), k2: dense scalar, same bit length
k_sparse = 1 << (n.bit_length() - 2)
k_dense = (1 << (n.bit_length() - 1)) - 3
# short scalar vs full-length scalar, exercises NAF length + ecdsa.py ks/kt
k_short = 1 << 8
k_full = (1 << (n.bit_length() - 1)) | 1

assert k_sparse < n and k_dense < n and k_full < n

print("curve=NIST256p order bits=%d iters=%d python=%s"
      % (n.bit_length(), ITERS, sys.version.split()[0]))
print("NAF lengths (loop iteration count is data dependent):")
for name, k in (("sparse", k_sparse), ("dense", k_dense),
                ("short", k_short), ("full", k_full)):
    print("  %-8s bits=%3d naf_len=%3d" % (name, k.bit_length(),
                                           len(AbstractPoint._naf(k))))

summ("sparse scalar (1 bit)", bench(k_sparse, ITERS))
summ("dense scalar (all bits)", bench(k_dense, ITERS))
summ("short scalar (9 bits)", bench(k_short, ITERS))
summ("full-length scalar", bench(k_full, ITERS))

# End-to-end deterministic signing: two messages only changes digest/k,
# illustrates whole-sign() timing spread, including RFC6979 retry-free path.
sk = SigningKey.generate(curve=NIST256p)
msgs = [bytes([i]) * 32 for i in range(8)]
sigs = []
t = []
for m in msgs:
    t0 = time.perf_counter_ns()
    sigs.append(sk.sign_deterministic(m, hashfunc=__import__("hashlib").sha256))
    t.append(time.perf_counter_ns() - t0)
print("sign_deterministic x8 msgs: min=%.1fus median=%.1fus max=%.1fus"
      % (min(t) / 1000, statistics.median(t) / 1000, max(t) / 1000))
