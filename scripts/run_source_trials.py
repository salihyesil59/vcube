"""Fit every candidate source and run RA-scrambled background trials for it.

Results go to data/processed/source_trials.npz and are used by the notebooks.

    python scripts/run_source_trials.py [--trials 5000] [--workers N]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from vcube.data import PROCESSED_DIR
from vcube.likelihood import PointSourceLLH
from vcube.pdfs import Model
from vcube.sources import CANDIDATES, source
from vcube.trials import p_value, run_trials, sigma_from_p

CHUNK = 500
OUT = PROCESSED_DIR / "source_trials.npz"

_model = None


def _init():
    global _model
    _model = Model()


def _source_job(args):
    name, n_trials, seed = args
    llh = PointSourceLLH(_model, *source(name))
    bkg = run_trials(llh, n_trials, seed=seed, progress=False)
    return name, llh.fit(), bkg["ts"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    # Trials run in chunks and are saved as they finish, so an interrupted run resumes.
    out = dict(np.load(OUT)) if OUT.exists() else {}
    jobs = []
    for i, (name, _, _) in enumerate(CANDIDATES):
        done = len(out.get(f"{name.replace(' ', '_')}__bkg_ts", []))
        for start in range(done, args.trials, CHUNK):
            jobs.append((name, min(CHUNK, args.trials - start), 1000 * i + start // CHUNK))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for fut in tqdm(as_completed([pool.submit(_source_job, j) for j in jobs]), total=len(jobs), desc="chunks"):
            name, fit, bkg_ts = fut.result()
            key = name.replace(" ", "_")
            out[f"{key}__fit"] = np.array([fit["ts"], fit["ns"], fit["gamma"]])
            out[f"{key}__bkg_ts"] = np.concatenate([out.get(f"{key}__bkg_ts", []), bkg_ts])
            np.savez(OUT, **out)

    for name, _, _ in CANDIDATES:
        key = name.replace(" ", "_")
        ts, ns, gamma = out[f"{key}__fit"]
        p = p_value(ts, out[f"{key}__bkg_ts"])
        print(f"{name:16s} TS={ts:6.2f} ns={ns:6.1f} gamma={gamma:.2f} p={p:.2e} ({sigma_from_p(p):.2f} sigma)")


if __name__ == "__main__":
    main()
