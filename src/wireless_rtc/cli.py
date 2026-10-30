import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from wireless_rtc.analysis.evaluate import evaluate, read_trace, save_evaluation
from wireless_rtc.analysis.formatting import ReportWriter
from wireless_rtc.network.delays import generate_delays
from wireless_rtc.run_paths import add_output_arguments, resolve_output


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
    add_output_arguments(suite)
    trace = commands.add_parser("evaluate-trace", help="Causal evaluation on completed RTT measurements")
    trace.add_argument("--trace", required=True)
    trace.add_argument("--config", default="configs/first_run.json")
    add_output_arguments(trace)
    qtrain = commands.add_parser("quantile-train", help="Train RTT quantile boosting; select settings on validation only")
    qtrain.add_argument("--data", nargs="+", required=True, help="Ordered measured CSV traces")
    qtrain.add_argument("--config", default="configs/quantile_boosting.json")
    add_output_arguments(qtrain)
    qtest = commands.add_parser("quantile-evaluate", help="Evaluate a frozen local quantile model on held-out traces")
    qtest.add_argument("--model", required=True, help="Trusted local model.pkl")
    qtest.add_argument("--data", nargs="+", required=True)
    qtest.add_argument("--partition", choices=["test", "all"], default="test",
                       help="test: held-out partition; all: entirely new traces")
    add_output_arguments(qtest)
    server = commands.add_parser("serve", help="Probe service; optional pretrained policy backend")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    checkpoints = server.add_mutually_exclusive_group()
    checkpoints.add_argument("--checkpoint",help="One fixed policy checkpoint")
    checkpoints.add_argument("--checkpoint-dir",help="Policy directory for automatic task switching")
    server.add_argument("--level", default="worlds/l/grasp_easy.json")
    server.add_argument("--delay", choices=["none","normal","pattern"], default="none", help="Optional extra policy response wait")
    server.add_argument("--delay-pattern", default="configs/delay_pattern.json", help="JSON phase pattern used with --delay pattern")
    server.add_argument("--delay-mean-ms", type=float, default=150)
    server.add_argument("--delay-std-ms", type=float, default=50)
    server.add_argument("--delay-max-ms", type=float, default=1000)
    server.add_argument("--delay-seed", type=int, default=42)
    probe = commands.add_parser("probe")
    probe.add_argument("--host", default="127.0.0.1")
    probe.add_argument("--port", type=int, default=8765)
    add_output_arguments(probe)
    probe.add_argument("--samples", type=int, default=100)
    probe.add_argument("--interval-ms", type=float, default=100)
    probe.add_argument("--payload-bytes", type=int, default=1024)
    probe.add_argument("--delay", choices=["none","fixed","random"], default="none")
    probe.add_argument("--seed", type=int, default=42)
    probe.add_argument("--wifi", action="store_true", help="Log Windows Wi-Fi telemetry in the background (no scans)")
    probe.add_argument("--wifi-interface", help="Wi-Fi adapter name; required if multiple adapters are present")
    probe.add_argument("--wifi-interval-ms", type=float, default=1000, help="Wi-Fi telemetry polling interval (default: 1000 ms)")
    characterize = commands.add_parser("network-characterize", help="Timed Wi-Fi characterization for Table 1 (fixed policy workload by default)")
    add_output_arguments(characterize)
    characterize.set_defaults(experiment="midterm", split="test")
    characterize.add_argument("--host", default="127.0.0.1")
    characterize.add_argument("--port", type=int, default=8765)
    characterize.add_argument("--duration-s", type=float, default=600)
    characterize.add_argument("--interval-ms", type=float, default=80, help="Pause after each response, not a fixed sampling period")
    characterize.add_argument("--timeout", type=float, default=10)
    characterize.add_argument("--workload", choices=["policy", "echo"], default="policy")
    characterize.add_argument("--task", default="cartpole_thrust")
    characterize.add_argument("--payload-bytes", type=int, default=1024, help="Echo workload only")
    characterize.add_argument("--seed", type=int, default=42)
    characterize.add_argument("--warmup", type=int, default=5)
    characterize.add_argument("--no-wifi", action="store_true", help="Disable radio collection for a transport-only diagnostic")
    characterize.add_argument("--wifi-interface")
    characterize.add_argument("--wifi-interval-ms", type=float, default=1000)
    characterize.add_argument("--notes", default="", help="Location or walking route and approximate lap time")
    network_summary = commands.add_parser("network-summary", help="Export Table 1 CSV and LaTeX from comparable completed runs")
    network_summary.add_argument("--config", default="configs/network_summary.json", help="Explicit input run selection")
    network_summary.add_argument("--experiment", default="midterm")
    network_summary.add_argument("--output-root", default="runs")
    network_summary.add_argument("--split", choices=["development", "train", "validation", "test"])
    network_summary.add_argument("--workload", choices=["policy", "echo"])
    network_summary.add_argument("--settings", nargs="+", choices=["strong", "weak", "stationary", "walking"],
                                 help="Include only these settings, in this order; use strong weak for the revised midterm")
    prediction_summary = commands.add_parser("prediction-summary", help="Table 2 on explicitly configured sessions; no collection or training")
    add_output_arguments(prediction_summary)
    prediction_summary.set_defaults(experiment="midterm", tag="table2", split="reports")
    prediction_summary.add_argument("--config", default="configs/midterm_prediction.json")
    prediction_summary.add_argument("--table1", help="Optional explicit Table 1 summary.json overriding the configured run selection")
    rollout_summary = commands.add_parser("rollout-summary", help="Tables 3/4 and internal diagnostics from completed rollout blocks")
    rollout_summary.add_argument("--config", default="configs/rollout_summary.json", help="Explicit rollout run selection")
    add_output_arguments(rollout_summary)
    rollout_summary.set_defaults(experiment="midterm", tag="tables3-4", split="reports")
    rollout_summary.add_argument("--settings", nargs="+", choices=["strong", "weak"])
    rollout_summary.add_argument("--allow-partial", action="store_true", help="Preliminary export with missing blocks explicitly recorded")
    sim = commands.add_parser("rtc", help="Actual pretrained RTC/Kinetix with independently injected delays")
    sim.add_argument("--config", default="configs/rtc_fixed.json")
    sim.add_argument("--checkpoint", required=True)
    add_output_arguments(sim)
    sim.add_argument("--view", action="store_true", help="Stream simulation frames on a loopback web viewer")
    sim.add_argument("--view-port", type=int, default=8080)
    sim.add_argument("--view-fps", type=float, default=10, help="Maximum rendered frames/second (default: 10)")
    sim.add_argument("--view-pace", action="store_true", help="With --view, limit stepping to the configured control rate")
    live = commands.add_parser("rtc-live", help="Paced local Kinetix simulator using a remote policy server")
    live.add_argument("--config", default="configs/rtc_remote.json")
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=8765)
    live.add_argument("--timeout", type=float, default=120, help="RPC timeout in seconds; includes startup warmup")
    live.add_argument("--block", type=int, help="Numbered rollout block (1-based); required by the midterm rollout config")
    add_output_arguments(live)
    live.add_argument("--view", action="store_true", help="Show simulation on a local browser viewer")
    live.add_argument("--view-port", type=int, default=8080)
    live.add_argument("--view-fps", type=float, default=10)
    live.set_defaults(view_pace=False)
    args = parser.parse_args(argv)
    if args.command == "rtc-live" and args.block is not None and args.tag is None and args.output is None:
        args.tag = f"block{args.block:02d}"
    if args.command not in {"serve", "network-summary"}:
        try:
            args.output = str(resolve_output(args))
        except ValueError as exc:
            parser.error(str(exc))
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
    if args.command == "rollout-summary":
        from wireless_rtc.analysis.rollout_summary import export_rollouts
        from wireless_rtc.analysis.run_selection import read_config
        try:
            _, config = read_config(args.config)
            export_rollouts(Path(args.output_root)/args.experiment/"rtc-live", args.output, args.settings,
                            args.allow_partial or config.get("allow_partial", False), args.config)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif args.command == "prediction-summary":
        from wireless_rtc.analysis.midterm_prediction import evaluate_table2
        from wireless_rtc.analysis.run_selection import read_config, data_path
        from wireless_rtc.network.characterization import select_network_runs
        try:
            config_path, config = read_config(args.config)
            root = Path(args.output_root) / args.experiment / "network-characterization"
            selection = None if args.table1 else select_network_runs(root,
                split=config.get("split", "test"), workload=config.get("workload", "policy"), selection_config=args.config)
            model = data_path(config_path, config, config["model"]) if config.get("model") else None
            evaluate_table2(args.table1, args.output, history=config["history"], control_ms=config["control_ms"], model=model, selection=selection)
        except (ValueError, OSError, RuntimeError) as exc:
            parser.error(str(exc))
    elif args.command == "network-summary":
        from types import SimpleNamespace
        from wireless_rtc.network.characterization import export_table
        from wireless_rtc.analysis.run_selection import read_config
        try:
            _, config = read_config(args.config)
            destination = resolve_output(SimpleNamespace(output=None, tag="table1", command="network-characterization",
                output_root=args.output_root, experiment=args.experiment, setting=None, split="reports"))
            root = Path(args.output_root) / args.experiment / "network-characterization"
            export_table(root, destination, args.split or config.get("split", "test"),
                         args.workload or config.get("workload", "policy"), args.settings, args.config)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif args.command == "network-characterize":
        from wireless_rtc.network.characterization import run_characterization
        if args.setting is None:
            parser.error("network-characterize requires --setting strong, weak, stationary or walking")
        print(f"Run output: {Path(args.output).resolve()}", flush=True)
        try:
            run_characterization(args.host, args.port, args.output, setting=args.setting, split=args.split,
                duration_s=args.duration_s, interval_ms=args.interval_ms, timeout=args.timeout,
                workload=args.workload, task=args.task, payload_bytes=args.payload_bytes, seed=args.seed,
                warmup=args.warmup, wifi=not args.no_wifi, wifi_interface=args.wifi_interface,
                wifi_interval_ms=args.wifi_interval_ms, notes=args.notes, run_metadata=metadata(vars(args)))
        except (ValueError, OSError, EOFError) as exc:
            parser.error(str(exc))
    elif args.command in {"quantile-train", "quantile-evaluate"}:
        from wireless_rtc.prediction.quantile_boosting import train_quantiles, evaluate_quantiles
        try:
            if args.command == "quantile-train":
                config = json.loads(Path(args.config).read_text(encoding="utf-8"))
                result = train_quantiles(args.data,config,args.output)
            else:
                result = evaluate_quantiles(args.model,args.data,args.output,args.partition)
        except (ValueError, RuntimeError, FileExistsError) as exc:
            parser.error(str(exc))
        print(f"Run output: {Path(result).resolve()}",flush=True)
    elif args.command == "serve":
        from wireless_rtc.server.service import serve
        response_delay = None
        if args.delay == "normal":
            from wireless_rtc.server.delay import NormalResponseDelay
            try:
                response_delay = NormalResponseDelay(args.delay_mean_ms,args.delay_std_ms,
                                                     args.delay_max_ms,args.delay_seed)
            except ValueError as exc:
                parser.error(str(exc))
        elif args.delay == "pattern":
            from wireless_rtc.server.delay import PatternResponseDelay
            try:
                pattern = json.loads(Path(args.delay_pattern).read_text(encoding="utf-8"))
                response_delay = PatternResponseDelay(pattern)
            except (OSError, ValueError) as exc:
                parser.error(f"Invalid delay pattern: {exc}")
        if response_delay is not None and not (args.checkpoint or args.checkpoint_dir):
            parser.error("Injected delay requires --checkpoint or --checkpoint-dir")
        policy = None
        if args.checkpoint:
            from wireless_rtc.simulation.kinetix import KinetixPolicy
            policy = KinetixPolicy(args.checkpoint, args.level)
        elif args.checkpoint_dir:
            from wireless_rtc.server.policies import PolicyRegistry
            try:
                policy = PolicyRegistry(args.checkpoint_dir)
            except ValueError as exc:
                parser.error(str(exc))
        if response_delay is not None and policy is None:
            parser.error("Injected delay requires --checkpoint or --checkpoint-dir")
        serve(args.host, args.port, policy, response_delay)
    elif args.command == "probe":
        print(f"Run output: {Path(args.output).resolve()}",flush=True)
        from wireless_rtc.network.probe import run_probe
        source = {"kind":"fixed","value_ms":0 if args.delay == "none" else 40}
        if args.delay == "random":
            source = {"kind":"uniform","low_ms":10,"high_ms":70}
        print(run_probe(args.host,args.port,args.output,args.samples,args.interval_ms,
                        args.payload_bytes,source,seed=args.seed,wifi=args.wifi,
                        wifi_interface=args.wifi_interface,wifi_interval_ms=args.wifi_interval_ms))
    elif args.command in {"rtc", "rtc-live"}:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.command == "rtc-live" and args.block is not None:
            config["block"] = args.block
        print(f"Run output: {Path(args.output).resolve()}",flush=True)
        def run(viewer=None):
            if args.command == "rtc":
                from wireless_rtc.simulation.kinetix import run_experiment
                return run_experiment(config,args.checkpoint,args.output,metadata(config),
                                      viewer=viewer,view_pace=args.view_pace)
            from wireless_rtc.client.live import run_remote
            run_metadata = metadata(config)
            run_metadata.update(experiment=args.experiment, setting=args.setting, split=args.split)
            return run_remote(config,args.host,args.port,args.output,run_metadata,
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
        print(f"Run output: {output.resolve()}",flush=True)
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
            writer = ReportWriter(f,fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    main()
