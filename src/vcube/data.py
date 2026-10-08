"""Loading of the IceTracks-DR2 event lists, uptime and instrument response files."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SEASONS = [
    "IC40", "IC59", "IC79",
    "IC86_I", "IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI",
    "IC86_VII", "IC86_VIII", "IC86_IX", "IC86_X", "IC86_XI",
]

EVENT_COLUMNS = {
    "run": "run",
    "event": "event",
    "subevent": "subevent",
    "MJD[days]": "mjd",
    "log10(E/GeV)": "logE",
    "AngErr[deg]": "ang_err",
    "RA[deg]": "ra",
    "Dec[deg]": "dec",
    "Azimuth[deg]": "azi",
    "Zenith[deg]": "zen",
}


def detector_config(season):
    """All IC86 seasons share one detector configuration and one set of IRFs."""
    return "IC86" if season.startswith("IC86") else season


def read_table(path):
    """Read a whitespace-separated release file whose header line starts with '#'."""
    with open(path) as fh:
        header = fh.readline().lstrip("#").split()
    return pd.read_csv(path, sep=r"\s+", comment="#", names=header, header=None)


@dataclass
class Season:
    name: str
    events: pd.DataFrame
    uptime: pd.DataFrame

    @property
    def config(self):
        return detector_config(self.name)

    @property
    def livetime_days(self):
        return float((self.uptime["MJD_stop[days]"] - self.uptime["MJD_start[days]"]).sum())


@cache
def load_season(name, raw_dir=RAW_DIR):
    events = read_table(raw_dir / "events" / f"{name}_exp.csv").rename(columns=EVENT_COLUMNS)
    # Angles in radians are what the analysis code works with.
    events["ra_rad"] = np.radians(events["ra"])
    events["dec_rad"] = np.radians(events["dec"])
    events["sin_dec"] = np.sin(events["dec_rad"])
    events["sigma_rad"] = np.radians(events["ang_err"])
    uptime = read_table(raw_dir / "uptime" / f"{name}_exp.csv")
    return Season(name, events, uptime)


def load_all(seasons=SEASONS, raw_dir=RAW_DIR):
    return [load_season(s, raw_dir) for s in seasons]


def all_events(seasons=SEASONS, raw_dir=RAW_DIR):
    """All events in one DataFrame, with a 'season' column."""
    return pd.concat(
        [s.events.assign(season=s.name) for s in load_all(seasons, raw_dir)], ignore_index=True
    )


@cache
def load_effective_area(config, raw_dir=RAW_DIR):
    df = read_table(raw_dir / "irfs" / f"{config}_effectiveArea.csv")
    return df.rename(columns={
        "log10(E_nu/GeV)_min": "logEnu_min",
        "log10(E_nu/GeV)_max": "logEnu_max",
        "Dec_nu_min[deg]": "dec_min",
        "Dec_nu_max[deg]": "dec_max",
        "A_Eff[cm^2]": "aeff_cm2",
    })
