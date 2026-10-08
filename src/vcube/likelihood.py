"""Unbinned maximum-likelihood point-source search (Braun et al. 2008).

    log L(ns, gamma) = sum_s sum_{i in s} log[ ns_s/N_s * S_i/B_i + (1 - ns_s/N_s) ]

with ns_s = ns * f_s(gamma) the share of signal events expected in season s.
Only events within `window` of the source have a non-negligible S_i; all other
events contribute log(1 - ns_s/N_s) each, which is summed analytically.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .pdfs import GAMMA_GRID, LOGE_EDGES, SINDEC_ENERGY_EDGES, angular_distance, bin_index, spatial_signal_pdf

GAMMA_BOUNDS = (GAMMA_GRID[0], GAMMA_GRID[-1])
NS_MAX = 1000.0


@dataclass
class Prepared:
    """Per-event quantities of one dataset (real or trial) around one source."""

    X: np.ndarray        # (n_win, n_gamma) S_i / B_i on GAMMA_GRID
    season: np.ndarray   # (n_win,) season index of each windowed event
    n_out: np.ndarray    # (n_seasons,) events per season outside the window
    n_total: np.ndarray  # (n_seasons,) all events per season


def _gamma_interp(gamma):
    x = (gamma - GAMMA_GRID[0]) / (GAMMA_GRID[1] - GAMMA_GRID[0])
    k = int(np.clip(np.floor(x), 0, len(GAMMA_GRID) - 2))
    return k, x - k


class PointSourceLLH:
    def __init__(self, model, ra, dec, window_deg=10.0):
        self.model = model
        self.ra, self.dec = ra, dec
        self.window = np.radians(window_deg)

        self.n_per_fluence = model.expected_events_per_fluence(dec)          # (S, G)
        self.season_frac = self.n_per_fluence / self.n_per_fluence.sum(axis=0)
        self.ratio_tables = [model.pdfs[c].energy_ratio_table(dec) for c in model.configs]

        ev = model.events
        band = np.abs(ev["dec_rad"] - dec) < self.window
        self.band_events = {k: v[band] for k, v in ev.items()}
        self.n_outside_band = model.n_events - np.bincount(
            self.band_events["season"], minlength=len(model.seasons)
        )

    # --- datasets ---------------------------------------------------------

    def prepare(self, events):
        """Compute S/B for band events (dict of arrays, e.g. real, scrambled or injected)."""
        n_seasons = len(self.model.seasons)
        n_band = np.bincount(events["season"], minlength=n_seasons)
        n_total = self.n_outside_band + n_band

        # Cheap right-ascension pre-cut before computing exact angular distances. From the
        # haversine formula, psi < w for |dec| <= D implies sin(dRA/2) < sin(w/2) / cos(D).
        x = np.sin(self.window / 2) / np.cos(min(abs(self.dec) + self.window, np.pi / 2))
        if x < 1:
            dra = np.abs(np.mod(events["ra_rad"] - self.ra + np.pi, 2 * np.pi) - np.pi)
            near = np.flatnonzero(dra < 2 * np.arcsin(x))
            events = {k: v[near] for k, v in events.items()}

        psi = angular_distance(events["ra_rad"], events["dec_rad"], self.ra, self.dec)
        win = psi < self.window
        e = {k: v[win] for k, v in events.items()}
        season = e["season"]
        psi = psi[win]
        n_out = n_total - np.bincount(season, minlength=n_seasons)

        X = np.empty((len(season), len(GAMMA_GRID)))
        config_idx = self.model.season_config[season]
        s_bin = bin_index(SINDEC_ENERGY_EDGES, e["sin_dec"])
        e_bin = bin_index(LOGE_EDGES, e["logE"])
        space_sig = spatial_signal_pdf(psi, e["sigma_rad"])
        for c, config in enumerate(self.model.configs):
            m = config_idx == c
            pdfs = self.model.pdfs[config]
            space_ratio = space_sig[m] / pdfs.background_space(e["sin_dec"][m])
            X[m] = space_ratio[:, None] * self.ratio_tables[c][:, s_bin[m], e_bin[m]].T
        return Prepared(X, season, n_out, n_total)

    def data(self):
        return self.prepare(self.band_events)

    # --- likelihood -------------------------------------------------------

    def logl(self, ns, gamma, prep):
        k, t = _gamma_interp(gamma)
        X = prep.X[:, k] * (1 - t) + prep.X[:, k + 1] * t
        frac = self.season_frac[:, k] * (1 - t) + self.season_frac[:, k + 1] * t
        w = ns * frac / prep.n_total                     # ns_s / N_s per season
        inside = np.log1p(w[prep.season] * (X - 1.0)).sum()
        outside = np.sum(prep.n_out * np.log1p(-w))
        return inside + outside

    def fit(self, prep=None):
        """Maximise log L over (ns >= 0, gamma). Returns dict with ts, ns, gamma."""
        prep = self.data() if prep is None else prep
        if len(prep.season) == 0:
            return {"ts": 0.0, "ns": 0.0, "gamma": np.nan}

        def neg(p):
            return -self.logl(p[0], p[1], prep)

        best = None
        for gamma0 in (2.0, 3.0):
            res = minimize(neg, x0=[1.0, gamma0], method="L-BFGS-B", bounds=[(0.0, NS_MAX), GAMMA_BOUNDS])
            if best is None or res.fun < best.fun:
                best = res
        ns, gamma = best.x
        ts = max(0.0, -2.0 * best.fun)
        return {"ts": ts, "ns": ns if ts > 0 else 0.0, "gamma": gamma if ts > 0 else np.nan}

    def ts_grid(self, ns_values, gamma_values, prep=None):
        """TS on a (gamma, ns) grid, for likelihood-landscape contour plots."""
        prep = self.data() if prep is None else prep
        return np.array([[2 * self.logl(n, g, prep) for n in ns_values] for g in gamma_values])

    # --- physics conversion ----------------------------------------------

    def flux_norm(self, ns, gamma):
        """phi0 [GeV^-1 cm^-2 s^-1] at 1 TeV for ns signal events with index gamma."""
        k, t = _gamma_interp(gamma)
        # The event yield changes by orders of magnitude with gamma: interpolate in log.
        log_total = np.log(self.n_per_fluence.sum(axis=0))
        return ns / np.exp(log_total[k] * (1 - t) + log_total[k + 1] * t)
