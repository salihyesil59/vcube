"""Pseudo-experiments: RA-scrambled background trials and signal injection."""

import numpy as np
from scipy import stats
from tqdm import tqdm

from .likelihood import _gamma_interp


def scrambled(llh, rng):
    """Band events with uniformly re-drawn right ascension (destroys any point source)."""
    ev = dict(llh.band_events)
    ev["ra_rad"] = rng.uniform(0, 2 * np.pi, len(ev["ra_rad"]))
    return ev


def rotate_from_source(ra_src, dec_src, psi, rng):
    """Directions at angular distance psi from the source, with random position angle."""
    phi = rng.uniform(0, 2 * np.pi, len(psi))
    sin_dec = np.sin(dec_src) * np.cos(psi) + np.cos(dec_src) * np.sin(psi) * np.cos(phi)
    dec = np.arcsin(np.clip(sin_dec, -1, 1))
    dra = np.arctan2(np.sin(phi) * np.sin(psi) * np.cos(dec_src), np.cos(psi) - np.sin(dec_src) * sin_dec)
    return np.mod(ra_src + dra, 2 * np.pi), dec


def inject(llh, mean_ns, gamma, rng):
    """Simulated signal events for an E^-gamma source, Poisson-distributed per season."""
    k, t = _gamma_interp(gamma)
    frac = llh.season_frac[:, k] * (1 - t) + llh.season_frac[:, k + 1] * t
    counts = rng.poisson(mean_ns * frac)
    parts = []
    for season, n in enumerate(counts):
        if n == 0:
            continue
        pdfs = llh.model.season_pdfs(season)
        logE, psf, angerr = pdfs.smearing.sample(llh.dec, gamma, pdfs.aeff, n, rng)
        ra, dec = rotate_from_source(llh.ra, llh.dec, psf, rng)
        parts.append({
            "ra_rad": ra, "dec_rad": dec, "sin_dec": np.sin(dec), "sigma_rad": angerr,
            "logE": logE, "season": np.full(n, season),
        })
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def trial(llh, rng, mean_ns=0.0, gamma=2.0):
    ev = scrambled(llh, rng)
    if mean_ns > 0:
        sig = inject(llh, mean_ns, gamma, rng)
        if sig is not None:
            ev = {k: np.concatenate([ev[k], sig[k]]) for k in ev}
    return llh.fit(llh.prepare(ev))


def run_trials(llh, n, seed=0, mean_ns=0.0, gamma=2.0, progress=True):
    rng = np.random.default_rng(seed)
    it = tqdm(range(n), disable=not progress, desc=f"trials ns={mean_ns:g}")
    out = [trial(llh, rng, mean_ns, gamma) for _ in it]
    return {k: np.array([o[k] for o in out]) for k in out[0]}


def p_value(ts, bkg_ts):
    """Empirical p-value; falls back to a chi2 fit of the tail when ts exceeds all trials."""
    bkg_ts = np.asarray(bkg_ts)
    n_above = np.sum(bkg_ts >= ts)
    if n_above > 0:
        return n_above / len(bkg_ts)
    eta = np.mean(bkg_ts > 0)
    df, _, scale = stats.chi2.fit(bkg_ts[bkg_ts > 0], floc=0)
    return eta * stats.chi2.sf(ts, df, scale=scale)


def sigma_from_p(p):
    """One-sided Gaussian significance."""
    return stats.norm.isf(p)


def sensitivity(llh, bkg_ts, gamma=2.0, ns_values=None, n_trials=200, seed=1, threshold=None, beta=0.9):
    """Mean injected ns giving TS > threshold in a fraction beta of trials.

    Default threshold is the background median (90% sensitivity). For a 5 sigma
    discovery potential pass threshold=<TS at 5 sigma> and beta=0.5.
    """
    threshold = np.median(bkg_ts) if threshold is None else threshold
    ns_values = np.arange(1, 41, 3) if ns_values is None else ns_values
    passing = []
    for i, ns in enumerate(ns_values):
        r = run_trials(llh, n_trials, seed=seed + i, mean_ns=ns, gamma=gamma, progress=False)
        passing.append(np.mean(r["ts"] > threshold))
    passing = np.maximum.accumulate(np.array(passing))
    ns90 = float(np.interp(beta, passing, ns_values)) if passing[-1] >= beta else np.nan
    return {"ns": ns90, "flux": llh.flux_norm(ns90, gamma), "ns_values": ns_values, "passing": passing}
