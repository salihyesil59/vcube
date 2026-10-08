"""Download the IceTracks-DR2 public data release from Harvard Dataverse.

By default this fetches all events, uptime lists, effective areas and the IC86
smearing matrix (~1 GB). Pass --all to also fetch the IC40/IC59/IC79 smearing
matrices (~2.6 GB more). The ICECAT-1 alert table (64 kB) is always fetched.

    python scripts/download_data.py [--all] [--dest data/raw]
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

DOI = "doi:10.7910/DVN/MMIIZA"
ICECAT_DOI = "doi:10.7910/DVN/SCRUCD"
ICECAT_FILE = "IceCube_Gold_Bronze_Tracks.csv"
API = "https://dataverse.harvard.edu/api"
DEFAULT_SMEARING = {"IC86_smearing.csv"}
# Dataverse rejects urllib's default User-Agent with 403.
HEADERS = {"User-Agent": "vcube-downloader/0.1"}


def urlopen(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS))


def list_files(doi=DOI):
    url = f"{API}/datasets/:persistentId/?persistentId={doi}"
    with urlopen(url) as resp:
        meta = json.load(resp)
    files = []
    for f in meta["data"]["latestVersion"]["files"]:
        df = f["dataFile"]
        # Dataverse converts uploaded CSVs into .tab; fetch the original CSV instead.
        name = df.get("originalFileName", df["filename"])
        files.append({
            "id": df["id"],
            "dir": f.get("directoryLabel", ""),
            "name": name,
            "original": "originalFileName" in df,
            "size": df.get("originalFileSize", df.get("filesize", 0)),
        })
    return files


def wanted(f, include_all):
    if f["name"].endswith(".png"):
        return False
    if f["name"].endswith("_smearing.csv"):
        return include_all or f["name"] in DEFAULT_SMEARING
    return True


def download(f, dest):
    out = dest / f["dir"] / f["name"]
    if out.exists() and out.stat().st_size == f["size"]:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/access/datafile/{f['id']}"
    if f["original"]:
        url += "?format=original"
    tmp = out.with_suffix(out.suffix + ".part")
    with urlopen(url) as resp, open(tmp, "wb") as fh, tqdm(
        total=f["size"] or None, unit="B", unit_scale=True, desc=f"{f['dir']}/{f['name']}", leave=False
    ) as bar:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            bar.update(len(chunk))
    tmp.replace(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="also download IC40/IC59/IC79 smearing matrices")
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()

    files = [f for f in list_files() if wanted(f, args.all)]
    files += [f | {"dir": "icecat1"} for f in list_files(ICECAT_DOI) if f["name"] == ICECAT_FILE]
    total = sum(f["size"] for f in files)
    print(f"Downloading {len(files)} files ({total / 1e9:.2f} GB) to {args.dest}")
    for f in tqdm(files, desc="files"):
        download(f, args.dest)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
