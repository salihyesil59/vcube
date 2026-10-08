"""All-sky scan: fit a point source at the centre of every HEALPix pixel."""

from concurrent.futures import ProcessPoolExecutor

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy_healpix import HEALPix
from tqdm import tqdm

from .likelihood import PointSourceLLH
from .pdfs import Model

_model = None


def pixel_centres(nside):
    """(ra, dec) in radians of all pixels in RING order."""
    lon, lat = HEALPix(nside=nside, order="ring").healpix_to_lonlat(np.arange(12 * nside**2))
    return lon.to_value("rad"), lat.to_value("rad")


def _init_worker(seasons, scramble_seed):
    global _model
    _model = Model(seasons)
    if scramble_seed is not None:
        _model.scramble(np.random.default_rng(scramble_seed))


def _scan_ring(args):
    dec, ras, window_deg = args
    llh = PointSourceLLH(_model, 0.0, dec, window_deg=window_deg)
    out = []
    for ra in ras:
        llh.ra = ra
        r = llh.fit()
        out.append((r["ts"], r["ns"], r["gamma"]))
    return out


def scan(model, nside=32, dec_range_deg=(-85.0, 85.0), window_deg=10.0, workers=None, scramble_seed=None):
    """TS, ns and gamma for every pixel. Pixels outside dec_range get NaN.

    With scramble_seed set, event right ascensions are randomised first: a
    background-only sky used to calibrate the hottest-spot significance.

    HEALPix pixels in RING order share their declination per ring, so the
    declination-band selection is done once per ring.
    """
    ra, dec = pixel_centres(nside)
    lo, hi = np.radians(dec_range_deg)
    result = np.full((3, len(ra)), np.nan)
    rings = [np.flatnonzero(dec == d) for d in np.unique(dec) if lo <= d <= hi]
    tasks = [(dec[idx[0]], ra[idx], window_deg) for idx in rings]

    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(model.seasons, scramble_seed)) as pool:
        for idx, out in zip(rings, tqdm(pool.map(_scan_ring, tasks), total=len(tasks), desc="rings")):
            result[:, idx] = np.array(out).T
    return {"ra": ra, "dec": dec, "ts": result[0], "ns": result[1], "gamma": result[2], "nside": nside}


def hottest(skymap, dec_range_deg=None):
    ts = skymap["ts"].copy()
    if dec_range_deg is not None:
        lo, hi = np.radians(dec_range_deg)
        ts[(skymap["dec"] < lo) | (skymap["dec"] > hi)] = np.nan
    i = int(np.nanargmax(ts))
    return {k: skymap[k][i] for k in ("ra", "dec", "ts", "ns", "gamma")} | {"pixel": i}


def plot_mollweide(values, nside, ax=None, cmap="magma", label="TS", vmax=None, n_grid=(900, 450)):
    """Mollweide projection of a RING-ordered HEALPix map, RA = 0 at the centre increasing to the left."""
    hp = HEALPix(nside=nside, order="ring")
    x = np.linspace(-np.pi, np.pi, n_grid[0])
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_grid[1])
    X, LAT = np.meshgrid(x, lat)
    img = np.asarray(values)[hp.lonlat_to_healpix(np.mod(-X, 2 * np.pi) * u.rad, LAT * u.rad)]

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6), subplot_kw={"projection": "mollweide"})
    mesh = ax.pcolormesh(X, LAT, img, cmap=cmap, shading="auto", vmin=0, vmax=vmax, rasterized=True)
    ticks = np.arange(-150, 151, 30)
    ax.set_xticks(np.radians(ticks))
    ax.set_xticklabels([f"{(-t) % 360}°" for t in ticks])
    ax.grid(alpha=0.3)
    plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, shrink=0.6, label=label)
    return ax


def mollweide_xy(ra, dec):
    """Axes coordinates of (ra, dec) [rad] on a plot_mollweide map."""
    return -(np.mod(np.asarray(ra) + np.pi, 2 * np.pi) - np.pi), np.asarray(dec)
