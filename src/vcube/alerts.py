"""ICECAT-1 gold/bronze track alerts and their cross-match with IceTracks-DR2."""

import numpy as np
import pandas as pd

from .data import RAW_DIR
from .pdfs import angular_distance


def load_icecat(raw_dir=RAW_DIR):
    df = pd.read_csv(raw_dir / "icecat1" / "IceCube_Gold_Bronze_Tracks.csv")
    df["ra_rad"] = np.radians(df["RA"])
    df["dec_rad"] = np.radians(df["DEC"])
    # 90% containment errors are asymmetric (RA errors in RA degrees); average them
    # into one circular radius on the sky.
    df["err90_deg"] = 0.5 * (
        0.5 * (df["RA_ERR_PLUS"] + df["RA_ERR_MINUS"]) * np.cos(df["dec_rad"])
        + 0.5 * (df["DEC_ERR_PLUS"] + df["DEC_ERR_MINUS"])
    )
    return df


def match_events(alerts, events):
    """Alerts whose (run, event) also appears in the DR2 track sample, with both reconstructions."""
    merged = alerts.merge(
        events[["run", "event", "season", "mjd", "ra", "dec", "ang_err", "logE"]],
        left_on=["RUNID", "EVENTID"], right_on=["run", "event"], how="inner",
    )
    merged["sep_deg"] = np.degrees(angular_distance(
        np.radians(merged["RA"]), np.radians(merged["DEC"]), np.radians(merged["ra"]), np.radians(merged["dec"]),
    ))
    return merged


def alerts_near(alerts, ra, dec, n_err=1.0):
    """Alerts whose 90% error region (scaled by n_err) contains the position (ra, dec) [rad]."""
    sep = np.degrees(angular_distance(alerts["ra_rad"], alerts["dec_rad"], ra, dec))
    return alerts.assign(sep_deg=sep)[sep <= n_err * alerts["err90_deg"]].sort_values("sep_deg")
