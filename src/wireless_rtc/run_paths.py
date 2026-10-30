"""Timestamped experiment destinations without overwriting previous runs."""
from datetime import datetime, timezone
from pathlib import Path
import re


def add_output_arguments(parser):
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output",help="Exact output directory (legacy; must not already exist)")
    output.add_argument("--tag",help="Run label; creates TAG-UTC_DATETIME under --output-root")
    parser.add_argument("--output-root",default="runs",help="Parent for timestamped runs (default: runs)")
    parser.add_argument("--experiment", help="Group new runs by experiment/command/setting/split")
    parser.add_argument("--setting", choices=["strong", "weak", "stationary", "walking"], help="Measured Wi-Fi setting (strong/weak are stationary locations)")
    parser.add_argument("--split", choices=["development", "train", "validation", "test"],
                        help="Purpose of this session; keep test runs separate")


def resolve_output(args, now=None):
    if args.output is not None:
        return Path(args.output)
    tag = args.tag or args.command
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",tag):
        raise ValueError("Run tag must be 1-64 letters/numbers/underscores/hyphens, starting with a letter or number")
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    root = Path(args.output_root)
    experiment = getattr(args, "experiment", None)
    if experiment is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", experiment):
            raise ValueError("Experiment must be a simple name of 1-64 letters/numbers/underscores/hyphens")
        category = {"network-characterize": "network-characterization"}.get(args.command, args.command)
        root = root / experiment / category
        if getattr(args, "setting", None):
            root = root / args.setting
        if getattr(args, "split", None):
            root = root / args.split
    elif getattr(args, "setting", None) or getattr(args, "split", None):
        raise ValueError("--setting and --split require --experiment")
    destination = root/f"{tag}-{stamp}"
    suffix = 2
    while destination.exists():
        destination = root/f"{tag}-{stamp}-{suffix:02d}"
        suffix += 1
    return destination
