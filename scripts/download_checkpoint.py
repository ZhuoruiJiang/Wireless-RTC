"""Download one official checkpoint, never the 60GiB expert/data directory."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

parser=argparse.ArgumentParser()
parser.add_argument("--epoch",type=int,default=24)
parser.add_argument("--level",default="grasp_easy")
parser.add_argument("--output",default="checkpoints")
args=parser.parse_args()
if args.epoch < 0 or not args.level.replace("_","").isalnum():
    parser.error("Invalid epoch or level")
name=f"worlds_l_{args.level}.pkl"
url=f"https://storage.googleapis.com/rtc-assets/bc/{args.epoch}/policies/{name}"
path=Path(args.output)/str(args.epoch)/"policies"/name
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists():
    raise SystemExit(f"Refusing to overwrite {path}")
with urllib.request.urlopen(url,timeout=60) as response:
    data=response.read()
path.write_bytes(data)
path.with_suffix(".json").write_text(json.dumps(dict(url=url,size_bytes=len(data),
    sha256=hashlib.sha256(data).hexdigest()),indent=2),encoding="utf-8")
print(path)
