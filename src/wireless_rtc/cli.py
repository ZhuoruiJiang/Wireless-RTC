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
    server.add_argument("--delay", choices=["none","normal"], default="none", help="Optional extra policy response wait")
    server.add_argument("--delay-mean-ms", type=float, default=150)
    server.add_argument("--delay-std-ms", type=float, default=50)
    server.add_argument("--delay-max-ms", type=float, default=1000)
    server.add_argument("--delay-seed", type=int, default=42)
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
    sim.add_argument("--view", action="store_true", help="Stream simulation frames on a loopback web viewer")
    sim.add_argument("--view-port", type=int, default=8080)
    sim.add_argument("--view-fps", type=float, default=10, help="Maximum rendered frames/second (default: 10)")
    sim.add_argument("--view-pace", action="store_true", help="With --view, limit stepping to the configured control rate")
    live = commands.add_parser("rtc-live", help="Paced local Kinetix simulator using a remote policy server")
    live.add_argument("--config", default="configs/rtc_remote.json")
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=8765)
    live.add_argument("--timeout", type=float, default=120, help="RPC timeout in seconds; includes startup warmup")
    live.add_argument("--output", required=True)
    live.add_argument("--view", action="store_true", help="Show simulation on a local browser viewer")
    live.add_argument("--view-port", type=int, default=8080)
    live.add_argument("--view-fps", type=float, default=10)
    live.set_defaults(view_pace=False)
    args = parser.parse_args(argv)
    if args.command in {"rtc", "rtc-live"}:
        if args.view_pace and not args.view:
            parser.error("--view-pace requires --view")
        if not 1 <= args.view_port <= 65535:
            parser.error("--view-port must be in [1,65535]")
        import math
        if not math.isfinite(args.view_fps) or not 0 < args.view_fps <= 60:
            parser.error("--view-fps must be in (0,60]")
        if args.command == "rtc-live" and (
                not math.isfinite(args.timeout) or args.timeout <= 0 or not 1 <= args.port <= 65535):
            parser.error("rtc-live requires a positive finite timeout and port in [1,65535]")
    if args.command == "serve":
        from wireless_rtc.server.service import serve
        response_delay = None
        if args.delay == "normal":
            from wireless_rtc.server.delay import NormalResponseDelay
            try:
                response_delay = NormalResponseDelay(args.delay_mean_ms,args.delay_std_ms,
                                                     args.delay_max_ms,args.delay_seed)
            except ValueError as exc:
                parser.error(str(exc))
        policy = None
        if args.checkpoint:
            from wireless_rtc.simulation.kinetix import KinetixPolicy
            policy = KinetixPolicy(args.checkpoint, args.level)
        if response_delay is not None and policy is None:
            parser.error("--delay normal requires --checkpoint; probe waits use the probe command's --delay")
        serve(args.host, args.port, policy, response_delay)
    elif args.command == "probe":
        from wireless_rtc.network.probe import run_probe
        source = {"kind":"fixed","value_ms":0 if args.delay == "none" else 40}
        if args.delay == "random":
            source = {"kind":"uniform","low_ms":10,"high_ms":70}
        print(run_probe(args.host,args.port,args.output,args.samples,args.interval_ms,
                        args.payload_bytes,source,seed=args.seed))
    elif args.command in {"rtc", "rtc-live"}:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        def run(viewer=None):
            if args.command == "rtc":
                from wireless_rtc.simulation.kinetix import run_experiment
                return run_experiment(config,args.checkpoint,args.output,metadata(config),
                                      viewer=viewer,view_pace=args.view_pace)
            from wireless_rtc.client.live import run_remote
            return run_remote(config,args.host,args.port,args.output,metadata(config),
                              timeout=args.timeout,viewer=viewer)
        if not args.view:
            run()
        else:
            import time
            from wireless_rtc.simulation.viewer import SimulationViewer
            try:
                viewer = SimulationViewer(args.view_port,args.view_fps)
            except OSError as exc:
                parser.error(f"Cannot open viewer port {args.view_port}: {exc}. Try another --view-port.")
            with viewer:
                location = "local browser" if args.command == "rtc-live" else "forward this port over SSH"
                print(f"RTC viewer: http://127.0.0.1:{viewer.port} ({location})", flush=True)
                exit_code = 0
                try:
                    run(viewer)
                    print("Run complete. Viewer remains open; press Ctrl+C to exit.", flush=True)
                except Exception as exc:
                    import sys
                    exit_code = 1
                    viewer.update(phase="Run failed",error=f"{type(exc).__name__}: {exc}")
                    print(f"Run failed: {exc}. Viewer remains open for inspection; Ctrl+C exits.",file=sys.stderr,flush=True)
                except KeyboardInterrupt:
                    print("Stopping RTC viewer.",flush=True)
                    return
                try:
                    while True:
                        time.sleep(.5)
                except KeyboardInterrupt:
                    print("Stopping RTC viewer.", flush=True)
                if exit_code:
                    raise SystemExit(exit_code)
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
