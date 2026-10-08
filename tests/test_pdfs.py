import numpy as np
from scipy import integrate

from vcube.irf import powerlaw_integral
from vcube.pdfs import angular_distance, spatial_signal_pdf
from vcube.trials import rotate_from_source


def test_spatial_pdf_normalised_on_sphere():
    for sigma_deg in (0.2, 1.0, 5.0, 30.0):
        sigma = np.radians(sigma_deg)
        total, _ = integrate.quad(lambda psi: spatial_signal_pdf(psi, sigma) * 2 * np.pi * np.sin(psi), 0, np.pi, points=[sigma])
        assert np.isclose(total, 1.0, rtol=1e-4), sigma_deg


def test_powerlaw_integral_matches_numeric():
    for gamma in (1.0, 2.0, 2.7):
        exact = powerlaw_integral(3.0, 4.0, gamma)
        numeric, _ = integrate.quad(lambda e: (e / 1e3) ** -gamma, 1e3, 1e4)
        assert np.isclose(exact, numeric, rtol=1e-8)


def test_rotation_keeps_requested_distance():
    rng = np.random.default_rng(0)
    for dec_src in np.radians([-80, -20, 0, 45, 85]):
        psi = rng.uniform(0, np.radians(10), 1000)
        ra, dec = rotate_from_source(1.0, dec_src, psi, rng)
        assert np.allclose(angular_distance(ra, dec, 1.0, dec_src), psi, atol=1e-8)
