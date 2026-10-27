import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from wireless_rtc.analysis.evaluate import evaluate, read_trace, save_evaluation
from wireless_rtc.network.delays import generate_delays


def metadata(config):
    def revision(path):
        result = subprocess.run(["git","-C",str(path),"rev-parse","HEAD"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(config=config, created_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), git_commit=revision("."),
                upstream_commit=revision("vendor/rtc-kinetix"),
                git_dirty=bool(subprocess.run(["git","status","--porcelain"],capture_output=True,text=True).stdout.strip()))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wireless RTC experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)
    suite = commands.add_parser("prediction-suite", help="Fixed/random/bursty synthetic prediction evaluation")
    suite.add_argument("--config", default="configs/first_run.json")
    suite.add_argument("--output", required=True)
    trace = commands.add_parser("evaluate-trace", help="Causal evaluation on completed RTT measurements")
    trace.add_argument("--trace", required=True)
    trace.add_argument("--config", default="configs/first_run.json")
    trace.add_argument("--output", required=True)
    server = commands.add_parser("serve", help="Probe service; optional pretrained policy backend")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--checkpoint")
    server.add_argument("--level", default="worlds/l/grasp_easy.json")
    probe = commands.add_parser("probe")
    probe.add_argument("--host", default="127.0.0.1")
    probe.add_argument("--port", type=int, default=8765)
    probe.add_argument("--output", required=True)
    probe.add_argument("--samples", type=int, default=100)
    probe.add_argument("--interval-ms", type=float, default=100)
    probe.add_argument("--payload-bytes", type=int, default=1024)
    probe.add_argument("--delay", choices=["none","fixed","random"], default="none")
    probe.add_argument("--seed", type=int, default=42)
    sim = commands.add_parser("rtc", help="Actual pretrained RTC/Kinetix with independently injected delays")
    sim.add_argument("--config", default="configs/rtc_fixed.json")
    sim.add_argument("--checkpoint", required=True)
    sim.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        from wireless_rtc.server.service import serve
        policy = None
        if args.checkpoint:
            from wireless_rtc.simulation.kinetix import KinetixPolicy
            policy = KinetixPolicy(args.checkpoint, args.level)
        serve(args.host, args.port, policy)
    elif args.command == "probe":
        from wireless_rtc.network.probe import run_probe
        source = {"kind":"fixed","value_ms":0 if args.delay == "none" else 40}
        if args.delay == "random":
            source = {"kind":"uniform","low_ms":10,"high_ms":70}
        print(run_probe(args.host,args.port,args.output,args.samples,args.interval_ms,
                        args.payload_bytes,source,seed=args.seed))
    elif args.command == "rtc":
        from wireless_rtc.simulation.kinetix import run_experiment
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        run_experiment(config, args.checkpoint, args.output, metadata(config))
    else:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        meta = metadata(config)
        if args.command == "evaluate-trace":
            sources = {"measured": read_trace(args.trace)}
            meta["trace_sha256"] = hashlib.sha256(Path(args.trace).read_bytes()).hexdigest()
            meta["trace_path"] = str(Path(args.trace).resolve())
        else:
            sources = {name:generate_delays(source, config["samples"],config["seed"])
                       for name,source in config["sources"].items()}
            meta["kind"] = "synthetic_prediction_evaluation"
        summaries = []
        for source,delays in sources.items():
            for name,predictor in config["predictors"].items():
                rows, metrics = evaluate(delays,predictor,config["control_ms"],
                                         config["warmup"],config["scoring_quantile"])
                save_evaluation(output/source/name, rows,metrics,meta)
                summaries.append(dict(source=source,predictor=name,**metrics))
        with (output/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    main()
