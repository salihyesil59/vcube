"""Tests that need the downloaded IceTracks-DR2 data (skipped otherwise)."""

import numpy as np
import pytest

from vcube.data import RAW_DIR

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "irfs" / "IC86_smearing.csv").exists(), reason="IceTracks-DR2 data not downloaded"
)


@pytest.fixture(scope="module")
def llh():
    from vcube.likelihood import PointSourceLLH
    from vcube.pdfs import Model

    model = Model(["IC86_I", "IC86_II", "IC86_III"])
    return PointSourceLLH(model, ra=np.radians(150.0), dec=np.radians(20.0))


def test_energy_pdf_normalised(llh):
    from vcube.pdfs import LOGE_EDGES

    pdfs = llh.model.pdfs["IC86"]
    for gamma in (2.0, 3.0):
        pdf = pdfs.smearing.energy_pdf(llh.dec, gamma, pdfs.aeff, LOGE_EDGES)
        assert np.isclose(np.sum(pdf * np.diff(LOGE_EDGES)), 1.0)


def test_softer_spectrum_has_lower_mean_energy(llh):
    from vcube.pdfs import LOGE_EDGES

    pdfs = llh.model.pdfs["IC86"]
    centres = 0.5 * (LOGE_EDGES[1:] + LOGE_EDGES[:-1])
    mean = [np.sum(centres * pdfs.smearing.energy_pdf(llh.dec, g, pdfs.aeff, LOGE_EDGES)) for g in (2.0, 3.0)]
    assert mean[0] > mean[1]


def test_background_ts_mostly_small(llh):
    from vcube.trials import run_trials

    r = run_trials(llh, 100, seed=0, progress=False)
    assert np.all(r["ts"] >= 0)
    assert np.median(r["ts"]) < 1.0


def test_injected_signal_is_recovered(llh):
    from vcube.trials import run_trials

    r = run_trials(llh, 50, seed=1, mean_ns=30, gamma=2.0, progress=False)
    assert np.median(r["ts"]) > 25
    assert abs(np.median(r["ns"]) - 30) < 8
    assert abs(np.median(r["gamma"]) - 2.0) < 0.3


def test_background_near_horizon_not_inflated(llh):
    # Regression: background bins straddling dec = -5 deg (selection boundary) made
    # scrambled skies near the equator look signal-like.
    from vcube.likelihood import PointSourceLLH
    from vcube.trials import run_trials

    equator = PointSourceLLH(llh.model, ra=np.radians(40.0), dec=0.0)
    r = run_trials(equator, 100, seed=2, progress=False)
    assert np.median(r["ts"]) < 0.3
