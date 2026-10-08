"""Signal and background PDFs for the unbinned point-source likelihood.

Signal:     S_i = vMF(psi_i; sigma_i) * S_E(logE_i | dec_src, gamma)
Background: B_i = P(sin dec_i) / 2pi * B_E(logE_i | sin dec_i)

Background PDFs are built from the data itself (it is dominated by atmospheric
muon neutrinos) and are uniform in right ascension. The signal energy PDF comes
from the smearing matrix weighted by the effective area and an E^-gamma spectrum.
"""

from dataclasses import dataclass, field

import numpy as np

from .data import SEASONS, detector_config, load_all
from .irf import SECONDS_PER_DAY, EffectiveArea, Smearing

GAMMA_GRID = np.round(np.arange(1.0, 4.0001, 0.1), 2)
LOGE_EDGES = np.arange(1.0, 9.51, 0.25)
# Background bins are uniform in declination (not sin dec) so that they stay narrow
# near the poles. In the southern sky the event selection raises the energy
# threshold steadily towards the south (~0.1 in log10 E per degree around the
# horizon), so the energy PDF needs fine declination bins there.
SINDEC_SPACE_EDGES = np.sin(np.radians(np.arange(-90, 91, 1.0)))
SINDEC_ENERGY_EDGES = np.sin(np.radians(np.concatenate([
    [-90], np.arange(-84, 6, 2), [10, 15, 20, 30, 40, 50, 60, 70, 90],
])))


def angular_distance(ra1, dec1, ra2, dec2):
    """Great-circle distance [rad] between points given in radians."""
    cos_psi = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.arccos(np.clip(cos_psi, -1.0, 1.0))


def spatial_signal_pdf(psi, sigma):
    """von Mises-Fisher (Kent) PDF on the sphere [sr^-1]; ~2D Gaussian for small sigma."""
    kappa = 1.0 / sigma**2
    return kappa / (2 * np.pi * -np.expm1(-2 * kappa)) * np.exp(kappa * (np.cos(psi) - 1.0))


def bin_index(edges, x):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


@dataclass
class ConfigPDFs:
    """PDFs for one detector configuration (all IC86 seasons share one)."""

    config: str
    aeff: EffectiveArea
    smearing: Smearing
    space_bkg: np.ndarray   # density in sin(dec), per SINDEC_SPACE_EDGES bin
    energy_bkg: np.ndarray  # (n_sindec, n_logE) density in logE per sin(dec) row
    _energy_ratio_cache: dict = field(default_factory=dict)

    @classmethod
    def build(cls, config, events):
        space, _ = np.histogram(events["sin_dec"], SINDEC_SPACE_EDGES, density=True)
        energy, _, _ = np.histogram2d(events["sin_dec"], events["logE"], [SINDEC_ENERGY_EDGES, LOGE_EDGES])
        norm = energy.sum(axis=1, keepdims=True) * np.diff(LOGE_EDGES)[None, :]
        energy = np.divide(energy, norm, out=np.zeros_like(energy), where=norm > 0)
        return cls(config, EffectiveArea.load(config), Smearing.load(config), space, energy)

    def background_space(self, sin_dec):
        """Background spatial PDF [sr^-1]."""
        return self.space_bkg[bin_index(SINDEC_SPACE_EDGES, sin_dec)] / (2 * np.pi)

    def energy_ratio_table(self, dec_src):
        """S_E / B_E on (GAMMA_GRID, sin dec bin, logE bin) for a source at dec_src."""
        key = (int(self.aeff.dec_bin(dec_src)), int(self.smearing.dec_bin(dec_src)))
        if key not in self._energy_ratio_cache:
            sig = np.array([self.smearing.energy_pdf(dec_src, g, self.aeff, LOGE_EDGES) for g in GAMMA_GRID])
            ratio = np.zeros((len(GAMMA_GRID),) + self.energy_bkg.shape)
            bkg = self.energy_bkg[None, :, :]
            sig = sig[:, None, :]
            ok = bkg > 0
            np.divide(sig, bkg, out=ratio, where=ok)
            # Where the background has no data but the signal does (high-energy tail),
            # use the largest ratio of that sin(dec) row instead of an infinite value.
            row_max = ratio.max(axis=2, keepdims=True)
            ratio = np.where(~ok & (sig > 0), row_max, ratio)
            self._energy_ratio_cache[key] = ratio
        return self._energy_ratio_cache[key]

    def expected_events_per_fluence(self, dec_src, livetime_days):
        """Expected signal events for phi0 = 1 GeV^-1 cm^-2 s^-1, per gamma in GAMMA_GRID."""
        seconds = livetime_days * SECONDS_PER_DAY
        return np.array([self.aeff.fluence_to_events(dec_src, g) * seconds for g in GAMMA_GRID])


class Model:
    """All seasons' events stacked into flat arrays plus per-configuration PDFs."""

    def __init__(self, seasons=SEASONS):
        loaded = load_all(seasons)
        self.seasons = [s.name for s in loaded]
        self.livetime_days = np.array([s.livetime_days for s in loaded])
        self.n_events = np.array([len(s.events) for s in loaded])
        self.configs = sorted({s.config for s in loaded})

        cols = ["ra_rad", "dec_rad", "sin_dec", "sigma_rad", "logE"]
        self.events = {c: np.concatenate([s.events[c].to_numpy() for s in loaded]) for c in cols}
        self.events["season"] = np.repeat(np.arange(len(loaded)), self.n_events)

        self.pdfs = {}
        for config in self.configs:
            mask = np.isin(self.events["season"], [i for i, s in enumerate(self.seasons) if detector_config(s) == config])
            self.pdfs[config] = ConfigPDFs.build(config, {c: v[mask] for c, v in self.events.items()})
        self.season_config = np.array([self.configs.index(detector_config(s)) for s in self.seasons])

    def scramble(self, rng):
        """Randomise all right ascensions in place (PDFs do not depend on RA)."""
        self.events["ra_rad"] = rng.uniform(0, 2 * np.pi, len(self.events["ra_rad"]))

    def season_pdfs(self, i):
        return self.pdfs[self.configs[self.season_config[i]]]

    def expected_events_per_fluence(self, dec_src):
        """(n_seasons, n_gamma) expected signal events for unit phi0 at dec_src."""
        return np.array([
            self.season_pdfs(i).expected_events_per_fluence(dec_src, t) for i, t in enumerate(self.livetime_days)
        ])
