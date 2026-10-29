"""Convert the published COMMECT MATLAB tables to a measured CSV without MATLAB.

The MAT reader uses its own environment so its NumPy requirements cannot change
the policy/experiment environment. Run from the repository root.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TABLES = ["cellular_latency_mobility", "cellular_latency_static"]


def convert(source, output, tables):
    import numpy as np
    from matio import load_from_mat
    data = load_from_mat(str(source))
    report = dict(source="https://zenodo.org/records/16919305",
                  source_mat_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  reader="mat-io==0.6.3",rtt_unit="ms",snr_unit="dB",sessions={},
                  timestamp_policy="milliseconds relative to first source timestamp within each session; original order retained",
                  filtering="no samples removed; failed RTTs remain explicit error rows")
    content = io.StringIO(newline="")
    writer = csv.DictWriter(content,fieldnames=["session_id","timestamp_ms","rtt_ms","snr_db","status"])
    writer.writeheader()
    for name in tables:
        frame = data[name]
        required = {"Timestamp","Round_Trip_Time","SNR"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name}: missing expected source columns")
        raw_time = frame["Timestamp"].to_numpy()
        if np.isnat(raw_time).any():
            raise ValueError(f"{name}: missing timestamp")
        nanos = raw_time.astype("datetime64[ns]").astype("int64")
        if (np.diff(nanos)<=0).any():
            raise ValueError(f"{name}: duplicate or out-of-order source timestamp")
        timestamp_ms = (nanos-nanos[0])/1e6
        rtts = frame["Round_Trip_Time"].to_numpy(dtype=float)
        snrs = frame["SNR"].to_numpy(dtype=float)
        failures = 0
        for t,rtt,snr in zip(timestamp_ms,rtts,snrs):
            good = bool(np.isfinite(rtt) and rtt>=0)
            failures += int(not good)
            writer.writerow(dict(session_id=name,timestamp_ms=float(t),rtt_ms=float(rtt) if good else "",
                                 snr_db=float(snr) if np.isfinite(snr) else "",status="ok" if good else "error"))
        report["sessions"][name] = dict(rows=len(frame),failed_rtt_rows=failures,missing_snr_rows=int((~np.isfinite(snrs)).sum()))
    output.parent.mkdir(parents=True,exist_ok=True)
    encoded = content.getvalue().encode("utf-8")
    if output.exists():
        if output.read_bytes() != encoded:
            raise ValueError(f"{output} already contains different data; choose another output path")
    else:
        with output.open("xb") as stream:
            stream.write(encoded)
    report["csv_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.with_suffix(".source.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Measured CSV ready: {output.resolve()}")
    for name,stats in report["sessions"].items():
        print(f"  {name}: {stats['rows']} source measurements")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"data/commect/raw/latency_data.mat")
    parser.add_argument("--output",type=Path,default=ROOT/"data/commect/measured.csv")
    parser.add_argument("--tables",nargs="+",choices=TABLES,default=TABLES)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        convert(args.input,args.output,args.tables)
        return
    if not args.input.is_file():
        subprocess.run([sys.executable,str(ROOT/"scripts/download_commect.py")],cwd=ROOT,check=True)
        if not args.input.is_file():
            parser.error(f"Input not found: {args.input}")
    environment = ROOT/".deps/commect-converter"
    python = environment/("Scripts/python.exe" if sys.platform=="win32" else "bin/python")
    if not python.is_file():
        subprocess.run(["uv","venv","--python","3.11",str(environment)],check=True)
    subprocess.run(["uv","pip","install","--python",str(python),"mat-io==0.6.3"],check=True)
    subprocess.run([str(python),str(Path(__file__).resolve()),"--worker","--input",str(args.input.resolve()),
                    "--output",str(args.output.resolve()),"--tables",*args.tables],check=True)


if __name__ == "__main__":
    main()
