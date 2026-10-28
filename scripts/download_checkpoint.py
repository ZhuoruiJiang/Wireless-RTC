"""Download official task checkpoints without fetching the expert/data directory."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import tempfile
from wireless_rtc.tasks import selected_tasks

def download(level,epoch,output,skip_existing=False):
    name = f"worlds_l_{level}.pkl"
    url = f"https://storage.googleapis.com/rtc-assets/bc/{epoch}/policies/{name}"
    path = Path(output)/str(epoch)/"policies"/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if skip_existing:
            print(f"Already present: {path}",flush=True)
            return path
        raise ValueError(f"Refusing to overwrite {path}; use --skip-existing to retain it")
    digest,size = hashlib.sha256(),0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".part",delete=False) as target:
            temporary = Path(target.name)
            with urllib.request.urlopen(url,timeout=120) as response:
                while chunk := response.read(1024*1024):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    path.with_suffix(".json").write_text(json.dumps(dict(url=url,size_bytes=size,
        sha256=digest.hexdigest()),indent=2),encoding="utf-8")
    print(path,flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch",type=int,default=24)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--level",help="One upstream level name (default: grasp_easy)")
    selection.add_argument("--config",help="Download every task selected by an rtc_remote config")
    parser.add_argument("--output",default="checkpoints")
    parser.add_argument("--skip-existing",action="store_true")
    args = parser.parse_args()
    try:
        levels = [task["name"] for task in selected_tasks(json.loads(Path(args.config).read_text()))] if args.config else [args.level or "grasp_easy"]
        if args.epoch < 0 or any(not level.replace("_","").isalnum() for level in levels):
            raise ValueError("Invalid epoch or level")
        for level in levels:
            download(level,args.epoch,args.output,args.skip_existing)
    except (ValueError,OSError) as exc:
        parser.exit(1,f"{exc}\n")


if __name__ == "__main__":
    main()
