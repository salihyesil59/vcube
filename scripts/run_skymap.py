"""All-sky point-source scan, optionally on RA-scrambled data.

    python scripts/run_skymap.py --nside 64
    python scripts/run_skymap.py --nside 32 --scrambles 20   # hottest-spot calibration
"""

import argparse

import numpy as np

from vcube.data import PROCESSED_DIR
from vcube.pdfs import Model
from vcube.skymap import hottest, scan

DEC_RANGE = (-85.0, 85.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--scrambles", type=int, default=0, help="number of scrambled skies instead of the data")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    model = Model()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if args.scrambles == 0:
        sky = scan(model, args.nside, DEC_RANGE, workers=args.workers)
        np.savez(PROCESSED_DIR / f"skymap_nside{args.nside}.npz", **sky)
        for hemi, rng in (("north", (-5, 85)), ("south", (-85, -5))):
            h = hottest(sky, rng)
            print(f"{hemi}: TS={h['ts']:.2f} ra={np.degrees(h['ra']):.2f} dec={np.degrees(h['dec']):.2f}")
        return

    out = PROCESSED_DIR / f"scrambled_hottest_nside{args.nside}.npz"
    done = dict(np.load(out)) if out.exists() else {"north": np.array([]), "south": np.array([])}
    start = len(done["north"])
    for seed in range(start, start + args.scrambles):
        sky = scan(model, args.nside, DEC_RANGE, workers=args.workers, scramble_seed=seed)
        done["north"] = np.append(done["north"], hottest(sky, (-5, 85))["ts"])
        done["south"] = np.append(done["south"], hottest(sky, (-85, -5))["ts"])
        np.savez(out, **done)
        print(f"scramble {seed}: north {done['north'][-1]:.2f} south {done['south'][-1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
