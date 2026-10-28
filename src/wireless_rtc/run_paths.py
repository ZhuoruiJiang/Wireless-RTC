"""Timestamped experiment destinations without overwriting previous runs."""
from datetime import datetime, timezone
from pathlib import Path
import re


def add_output_arguments(parser):
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output",help="Exact output directory (legacy; must not already exist)")
    output.add_argument("--tag",help="Run label; creates TAG-UTC_DATETIME under --output-root")
    parser.add_argument("--output-root",default="runs",help="Parent for timestamped runs (default: runs)")


def resolve_output(args, now=None):
    if args.output is not None:
        return Path(args.output)
    tag = args.tag or args.command
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",tag):
        raise ValueError("Run tag must be 1-64 letters/numbers/underscores/hyphens, starting with a letter or number")
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    root = Path(args.output_root)
    destination = root/f"{tag}-{stamp}"
    suffix = 2
    while destination.exists():
        destination = root/f"{tag}-{stamp}-{suffix:02d}"
        suffix += 1
    return destination
