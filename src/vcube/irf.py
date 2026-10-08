"""Instrument response: effective areas and smearing matrices.

The effective area gives the expected event rate for a flux model. The smearing
matrix maps a true (E_nu, dec_nu) bin onto reconstructed (log10 E, PSF, AngErr)
bins; it is used to build the signal energy PDF and to inject simulated signal
events.

Only the IC86 smearing matrix is required: IC40/IC59/IC79 fall back to it when
their own matrix has not been downloaded (an approximation that only affects
~3 of the 14 years).
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from .data import PROCESSED_DIR, RAW_DIR, load_effective_area, read_table

SMEARING_COLUMNS = [
    "logEnu_min", "logEnu_max", "dec_min", "dec_max", "logE_min", "logE_max",
    "psf_min", "psf_max", "angerr_min", "angerr_max", "frac",
]

# Reference energy for flux normalisations: dN/dE = phi0 * (E / 1 TeV)^-gamma.
E0_GEV = 1e3
SECONDS_PER_DAY = 86400.0


def powerlaw_integral(logE_min, logE_max, gamma, e0=E0_GEV):
    """Integral of (E / e0)^-gamma dE over [10^logE_min, 10^logE_max] GeV."""
    e1, e2 = 10.0 ** np.asarray(logE_min), 10.0 ** np.asarray(logE_max)
    if np.isclose(gamma, 1.0):
        return e0 * np.log(e2 / e1)
    return e0 / (1.0 - gamma) * ((e2 / e0) ** (1.0 - gamma) - (e1 / e0) ** (1.0 - gamma))


@dataclass
class EffectiveArea:
    logEnu_edges: np.ndarray  # (nE + 1,)
    dec_edges: np.ndarray     # (nDec + 1,) radians
    aeff_cm2: np.ndarray      # (nE, nDec)

    @classmethod
    def load(cls, config):
        df = load_effective_area(config)
        logE_edges = np.union1d(df["logEnu_min"], df["logEnu_max"])
        dec_edges = np.radians(np.union1d(df["dec_min"], df["dec_max"]))
        i = np.searchsorted(logE_edges, df["logEnu_min"])
        j = np.searchsorted(dec_edges, np.radians(df["dec_min"]))
        aeff = np.zeros((len(logE_edges) - 1, len(dec_edges) - 1))
        aeff[i, j] = df["aeff_cm2"]
        return cls(logE_edges, dec_edges, aeff)

    def dec_bin(self, dec):
        return np.clip(np.searchsorted(self.dec_edges, dec, side="right") - 1, 0, len(self.dec_edges) - 2)

    def fluence_to_events(self, dec, gamma, logE_min=None, logE_max=None):
        """Expected events per unit fluence phi0 * livetime [GeV^-1 cm^-2] for a source at dec.

        Optionally restricted to a true-energy range (used to weight smearing bins).
        """
        lo = self.logEnu_edges[:-1] if logE_min is None else np.clip(self.logEnu_edges[:-1], logE_min, logE_max)
        hi = self.logEnu_edges[1:] if logE_max is None else np.clip(self.logEnu_edges[1:], logE_min, logE_max)
        return float(np.sum(self.aeff_cm2[:, self.dec_bin(dec)] * powerlaw_integral(lo, hi, gamma)))


def _smearing_cache_path(config):
    return PROCESSED_DIR / f"{config}_smearing.npz"


def preprocess_smearing(config, raw_dir=RAW_DIR, chunksize=500_000):
    """Convert the (large) smearing CSV into a compact .npz without empty bins."""
    import pandas as pd

    path = raw_dir / "irfs" / f"{config}_smearing.csv"
    parts = []
    reader = pd.read_csv(
        path, sep=r"\s+", comment="#", names=SMEARING_COLUMNS, header=None,
        dtype=np.float64, chunksize=chunksize,
    )
    for chunk in reader:
        parts.append(chunk[chunk["frac"] > 0].to_numpy())
    table = np.concatenate(parts)
    out = _smearing_cache_path(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{c: table[:, k] for k, c in enumerate(SMEARING_COLUMNS)})
    return out


@dataclass
class Smearing:
    config: str
    cols: dict  # column name -> 1D array, one entry per non-empty 5D bin

    @classmethod
    @cache
    def load(cls, config):
        path = _smearing_cache_path(config)
        if not path.exists():
            if not (RAW_DIR / "irfs" / f"{config}_smearing.csv").exists():
                if config == "IC86":
                    raise FileNotFoundError(f"{path} missing; run scripts/download_data.py first")
                return cls.load("IC86")
            preprocess_smearing(config)
        with np.load(path) as f:
            cols = {c: f[c] for c in SMEARING_COLUMNS}
        cols["dec_min"] = np.radians(cols["dec_min"])
        cols["dec_max"] = np.radians(cols["dec_max"])
        return cls(config, cols)

    @property
    def dec_edges(self):
        return np.union1d(self.cols["dec_min"], self.cols["dec_max"])

    def dec_bin(self, dec):
        edges = self.dec_edges
        return np.clip(np.searchsorted(edges, dec, side="right") - 1, 0, len(edges) - 2)

    def rows_for_dec(self, dec):
        """Boolean mask of the rows belonging to the true-declination bin containing dec."""
        edges = self.dec_edges
        k = self.dec_bin(dec)
        return np.isclose(self.cols["dec_min"], edges[k])

    def row_weights(self, mask, dec, gamma, aeff):
        """Relative signal weight of each selected row for an E^-gamma source at dec.

        weight = fraction-within-(E_nu, dec_nu)-bin * expected events in that E_nu bin.
        """
        c = {k: v[mask] for k, v in self.cols.items()}
        enu_bins, inverse = np.unique(np.stack([c["logEnu_min"], c["logEnu_max"]], axis=1), axis=0, return_inverse=True)
        per_bin = np.array([aeff.fluence_to_events(dec, gamma, lo, hi) for lo, hi in enu_bins])
        return c, c["frac"] * per_bin[inverse.ravel()]

    def energy_pdf(self, dec, gamma, aeff, logE_edges):
        """Signal PDF of reconstructed log10(E/GeV) on logE_edges (density per unit logE)."""
        c, w = self.row_weights(self.rows_for_dec(dec), dec, gamma, aeff)
        # Spread each row's weight uniformly over its reco-energy bin.
        lo = np.maximum(c["logE_min"][:, None], logE_edges[None, :-1])
        hi = np.minimum(c["logE_max"][:, None], logE_edges[None, 1:])
        overlap = np.clip(hi - lo, 0, None) / (c["logE_max"] - c["logE_min"])[:, None]
        hist = (w[:, None] * overlap).sum(axis=0)
        return hist / (hist.sum() * np.diff(logE_edges))

    def sample(self, dec, gamma, aeff, n, rng):
        """Draw n signal events (logE, psf [rad], angerr [rad]) for an E^-gamma source at dec."""
        c, w = self.row_weights(self.rows_for_dec(dec), dec, gamma, aeff)
        idx = rng.choice(len(w), size=n, p=w / w.sum())

        def uniform(name):
            lo, hi = c[f"{name}_min"][idx], c[f"{name}_max"][idx]
            return lo + (hi - lo) * rng.random(n)

        return uniform("logE"), np.radians(uniform("psf")), np.radians(uniform("angerr"))
