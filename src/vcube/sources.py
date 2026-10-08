"""Candidate neutrino sources and a simple stacking likelihood."""

import numpy as np
from scipy.optimize import minimize

from .likelihood import GAMMA_BOUNDS, NS_MAX, PointSourceLLH, _gamma_interp

# (name, RA [deg], Dec [deg]) — J2000 positions.
CANDIDATES = [
    ("NGC 1068", 40.67, -0.01),
    ("TXS 0506+056", 77.36, 5.69),
    ("NGC 4151", 182.64, 39.41),
    ("NGC 3079", 150.49, 55.68),
    ("CGCG 420-015", 73.36, 4.06),
    ("PKS 1424+240", 216.75, 23.80),
    ("GB6 J1542+6129", 235.74, 61.50),
    ("Mrk 421", 166.11, 38.21),
    ("Mrk 501", 253.47, 39.76),
    ("Crab Nebula", 83.63, 22.01),
    ("Cyg A", 299.87, 40.73),
    ("M87", 187.71, 12.39),
]

# Seyfert galaxies used as an illustrative stacking catalogue (X-ray bright, northern sky).
SEYFERT_STACK = ["NGC 1068", "NGC 4151", "NGC 3079", "CGCG 420-015"]


def source(name):
    for n, ra, dec in CANDIDATES:
        if n == name:
            return np.radians(ra), np.radians(dec)
    raise KeyError(name)


class StackingLLH:
    """Joint likelihood of several sources sharing one ns and gamma.

    Each source gets a share of ns proportional to its expected event yield
    (equal intrinsic flux weighting). Signal windows are assumed not to overlap.
    """

    def __init__(self, model, names, window_deg=10.0):
        self.names = names
        self.llhs = [PointSourceLLH(model, *source(n), window_deg=window_deg) for n in names]
        yields = np.array([l.n_per_fluence.sum(axis=0) for l in self.llhs])  # (n_src, G)
        self.source_frac = yields / yields.sum(axis=0)

    def fit(self, preps=None):
        preps = [l.data() for l in self.llhs] if preps is None else preps

        def neg(p):
            ns, gamma = p
            k, t = _gamma_interp(gamma)
            frac = self.source_frac[:, k] * (1 - t) + self.source_frac[:, k + 1] * t
            # Events outside every window are counted once per source: correct to first
            # order since each source's term is ~ -ns_j f_s / N_s for those events.
            return -sum(l.logl(ns * f, gamma, prep) for l, f, prep in zip(self.llhs, frac, preps))

        best = min(
            (minimize(neg, x0=[1.0, g0], method="L-BFGS-B", bounds=[(0.0, NS_MAX), GAMMA_BOUNDS]) for g0 in (2.0, 3.0)),
            key=lambda r: r.fun,
        )
        ts = max(0.0, -2.0 * best.fun)
        return {"ts": ts, "ns": best.x[0] if ts > 0 else 0.0, "gamma": best.x[1] if ts > 0 else np.nan}
