"""Fetch the fixed public measured dataset; no network or model simulation."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


URL = "https://zenodo.org/records/16919305/files/dataset.zip?download=1"
EXPECTED_MD5 = "bf80eee0f22dd5bcee2ad092031cffd1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("data/commect"))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    archive = args.output/"dataset.zip"
    if archive.exists():
        data = archive.read_bytes()
    else:
        with urlopen(URL,timeout=60) as response:
            data = response.read()
    if hashlib.md5(data).hexdigest() != EXPECTED_MD5:
        raise ValueError("COMMECT archive checksum differs from the published fixed version")
    if not archive.exists():
        with archive.open("xb") as stream:
            stream.write(data)
    # Write one exact member, not arbitrary paths supplied by an archive.
    with ZipFile(archive) as source:
        latency = source.read("latency_data.mat")
    destination = args.output/"raw"/"latency_data.mat"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != latency:
            raise ValueError("Existing latency MAT file differs; refusing to overwrite it")
    else:
        with destination.open("xb") as stream:
            stream.write(latency)
    manifest = dict(source="https://zenodo.org/records/16919305",url=URL,md5=EXPECTED_MD5,
                    sha256=hashlib.sha256(data).hexdigest(),measurement_type="real cellular/satellite measurements",
                    note="Export a cellular MATLAB table to CSV before training; not Wi-Fi control data")
    (args.output/"source.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Verified archive and latency table: {args.output.resolve()}")


if __name__ == "__main__":
    main()
