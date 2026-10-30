"""Timed, reproducible Wi-Fi characterization and Table 1 export.

Policy mode repeats one valid reset observation, not a closed-loop rollout.
Echo mode is a separate measurement target and is never pooled with policy mode.
"""
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from wireless_rtc.analysis.formatting import ReportWriter
from wireless_rtc.client.live import validate_actions
from wireless_rtc.network.transport import RPCClient
from wireless_rtc.network.wifi import WIFI_FIELDS, WifiSampler
from wireless_rtc.prediction.rolling import quantile
from wireless_rtc.tasks import TASKS


FIELDS = ["index", "wall_time_ns", "request_start_monotonic_ns", "request_end_monotonic_ns",
          "status", "actual_ms", "failed_elapsed_ms", "server_ms", "inference_ms",
          "request_bytes", "response_bytes", "error"] + WIFI_FIELDS + ["wifi_age_ms"]


def _size(message):
    return len(json.dumps(message, allow_nan=False).encode("utf-8")) + 1


def prepare_workload(client, workload, task, payload_bytes, seed, warmup):
    """Warm up before starting the measurement clock; reject injected policy waits."""
    if workload == "echo":
        fields = dict(payload="x" * payload_bytes, hold_ms=0)
        for _ in range(warmup):
            client.request("probe", **fields)
        return "probe", fields, lambda response: None, dict(workload="echo", payload_bytes=payload_bytes)

    from wireless_rtc.simulation.kinetix import KinetixSimulator
    sim = KinetixSimulator(TASKS[task], seed)
    selected, _ = client.request("select_task", task=task)
    if selected.get("response_delay", {}).get("kind") != "none":
        raise ValueError("Characterization requires the policy server started with --delay none")
    expected = dict(task=task, level_sha256=sim.level_sha256,
                    observation_dim=sim.observation_dim, action_dim=sim.action_dim)
    if any(selected.get(key) != value for key, value in expected.items()):
        raise ValueError("Client simulator and server policy do not match")
    obs, _ = sim.reset(sim.jax.random.key(seed))
    observation = sim.np.asarray(obs).tolist()
    horizon = selected["horizon"]
    if type(horizon) is not int or horizon < 2:
        raise ValueError("Policy horizon must be at least two")
    initial, _ = client.request("initialize", task=task, observation=observation, seed=seed)
    previous = validate_actions(initial, 0, horizon, sim.action_dim, task)
    fields = dict(task=task, observation=observation, previous_actions=previous,
                  observation_tick=0, predicted_steps=1, overlap_steps=min(4, horizon),
                  method="wireless_rtc")

    def validate(response):
        validate_actions(response, 0, horizon, sim.action_dim, task)
        if response.get("injected_delay_ms") != 0:
            raise ValueError("Server returned an injected policy delay")

    for _ in range(warmup):
        response, _ = client.request("infer", **fields)
        validate(response)
    detail = dict(workload="policy", task=task, seed=seed, horizon=horizon,
                  level_sha256=sim.level_sha256, checkpoint_sha256=selected.get("checkpoint_sha256"),
                  observation_sha256=hashlib.sha256(json.dumps(observation).encode()).hexdigest(),
                  predicted_steps=1, overlap_steps=fields["overlap_steps"],
                  note="Fixed reset observation and initial previous chunk; not a control rollout")
    return "infer", fields, validate, detail


def run_characterization(host, port, output, *, setting, split, duration_s=600,
                         interval_ms=80, timeout=10, workload="policy", task="cartpole_thrust",
                         payload_bytes=1024, seed=42, warmup=5, wifi=True,
                         wifi_interface=None, wifi_interval_ms=1000, notes="", run_metadata=None):
    for name, value in [("duration", duration_s), ("timeout", timeout), ("interval", interval_ms)]:
        if not math.isfinite(value) or (value < 0 if name == "interval" else value <= 0):
            raise ValueError(f"Invalid {name}")
    if setting not in {"strong", "weak", "stationary", "walking"} or split not in {"development", "train", "validation", "test"}:
        raise ValueError("Choose a setting and session split")
    if workload not in {"policy", "echo"} or task not in TASKS:
        raise ValueError("Unknown workload or task")
    if type(warmup) is not int or warmup < 1 or not 0 <= payload_bytes <= 1000000 or not 0 <= seed < 2**32:
        raise ValueError("Invalid warmup, payload size or seed")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid server port")
    output = Path(output)
    sampler = WifiSampler(output / "wifi.csv", wifi_interface, wifi_interval_ms) if wifi else None
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(kind="network_characterization", setting=setting, split=split,
                    created_utc=datetime.now(timezone.utc).isoformat(), host=host, port=port,
                    requested_duration_s=duration_s, interval_ms=interval_ms, timeout_s=timeout,
                    response_duration_clock="perf_counter_ns",
                    warmup_requests=warmup, wifi_enabled=wifi, wifi_interface=wifi_interface,
                    wifi_interval_ms=wifi_interval_ms, notes=notes, status="preparing",
                    workload=dict(workload=workload), metadata=run_metadata)

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_manifest()
    client = None
    started = None
    try:
        print("Preparing workload and warming policy; stay at the route start until RECORDING appears.", flush=True)
        client = RPCClient(host, port, max(120, timeout))
        operation, fields, validate, detail = prepare_workload(client, workload, task, payload_bytes, seed, warmup)
        manifest["workload"] = detail
        client.socket.settimeout(timeout)
        with (sampler if sampler is not None else nullcontext()) as telemetry, (
                (output / "trace.csv").open("w", newline="", encoding="utf-8")) as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            started = time.monotonic_ns()
            manifest.update(status="recording", measurement_start_ns=started)
            save_manifest()
            deadline = started + int(duration_s * 1e9)
            index = successful = failed = 0
            next_progress = started
            print(f"RECORDING {setting}: {duration_s:g} seconds. Press Ctrl+C to preserve a partial run.", flush=True)
            if telemetry is not None and telemetry.snapshot().get("wifi_status") != "ok":
                print("Wi-Fi telemetry is unavailable; errors will be retained and RSSI may be missing.", flush=True)
            while time.monotonic_ns() < deadline:
                row = dict(index=index, wall_time_ns=time.time_ns(), status="interrupted")
                if telemetry is not None:
                    row.update(telemetry.snapshot())
                row["request_start_monotonic_ns"] = time.monotonic_ns()
                try:
                    if client is None:
                        client = RPCClient(host, port, timeout)
                    row["request_bytes"] = _size(dict(request_id=client.counter, operation=operation, **fields))
                    # Windows monotonic_ns can advance in coarse (~16 ms) ticks.
                    # perf_counter_ns is the high-resolution duration clock.
                    rpc_start = time.perf_counter_ns()
                    response, _ = client.request(operation, **fields)
                    validate(response)
                    elapsed = (time.perf_counter_ns() - rpc_start) / 1e6
                    row.update(status="ok", actual_ms=elapsed, server_ms=response.get("server_ms"),
                               inference_ms=response.get("inference_ms"), response_bytes=_size(response))
                    successful += 1
                except (OSError, EOFError) as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                               failed_elapsed_ms=(time.monotonic_ns()-row["request_start_monotonic_ns"]) / 1e6)
                    failed += 1
                    if client is not None:
                        client.close()
                        client = None
                except ValueError as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    row["request_end_monotonic_ns"] = time.monotonic_ns()
                    writer.writerow(row)
                    stream.flush()
                index += 1
                now = time.monotonic_ns()
                if now >= next_progress:
                    print(f"{(now-started)/1e9:.0f}/{duration_s:g} s | responses {successful} | errors {failed}", flush=True)
                    next_progress = now + 30_000_000_000
                time.sleep(min(interval_ms / 1000, max(0, (deadline-time.monotonic_ns())/1e9)))
            manifest["status"] = "complete"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        print("Stopped; partial measurements preserved (excluded from final table export).", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if client is not None:
            client.close()
        manifest["measurement_end_ns"] = time.monotonic_ns()
        if started is not None:
            manifest["elapsed_s"] = (manifest["measurement_end_ns"]-started)/1e9
        save_manifest()
        if (output / "trace.csv").exists():
            summary = summarize_run(output)
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Summary: {summary['completed_requests']} responses, {summary['failed_requests']} errors, "
                  f"median RSSI {summary['median_rssi_dbm']} dBm, p95 {summary['response_p95_ms']} ms", flush=True)
    print(f"Saved: {output.resolve()}", flush=True)
    return output


def _rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _measurements(directory, manifest):
    rows = _rows(directory / "trace.csv")
    delays = [float(r["actual_ms"]) for r in rows if r["status"] == "ok"]
    if any(not math.isfinite(v) or v < 0 for v in delays):
        raise ValueError(f"Invalid response duration in {directory}")
    start, end = manifest.get("measurement_start_ns", 0), manifest.get("measurement_end_ns", 0)
    polls = [r for r in _rows(directory / "wifi.csv")
             if start <= int(r["wifi_available_monotonic_ns"]) <= end]
    signals = [float(r["wifi_rssi_dbm"]) for r in polls
               if r["wifi_status"] == "ok" and r.get("wifi_rssi_dbm")]
    if any(not math.isfinite(v) for v in signals):
        raise ValueError(f"Invalid RSSI in {directory}")
    return rows, delays, polls, signals


def summarize_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows, delays, polls, signals = _measurements(directory, manifest)
    return dict(setting=manifest["setting"], split=manifest["split"], status=manifest["status"],
                workload=manifest["workload"], requests=len(rows), completed_requests=len(delays),
                failed_requests=sum(r["status"] != "ok" for r in rows),
                median_rssi_dbm=statistics.median(signals) if signals else None,
                response_p50_ms=quantile(delays, .5) if delays else None,
                response_p95_ms=quantile(delays, .95) if delays else None,
                wifi_polls=len(polls), valid_rssi_polls=len(signals),
                note="Percentiles condition on successful responses; RSSI uses independent poll records, not request joins")


def select_network_runs(root, split="test", workload="policy", settings=None, selection_config=None):
    """Pool requests/polls within each setting; never pool different workloads."""
    root = Path(root)
    from wireless_rtc.analysis.run_selection import load_run_selection, selection_audit
    explicit = load_run_selection(selection_config, settings) if selection_config is not None else None
    if explicit is not None:
        settings = explicit["settings"]
    labels = {"strong": "Strong Wi-Fi", "weak": "Weak Wi-Fi", "stationary": "Stationary", "walking": "Walking route"}
    settings = list(dict.fromkeys(settings if settings is not None else labels))
    if not settings or any(setting not in labels for setting in settings):
        raise ValueError("Choose supported settings to export")
    mapping_path = root / "session_map.json"
    mapping = {}
    if explicit is None and mapping_path.exists():
        document = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping = document.get("settings")
        if document.get("schema_version") != 1 or not isinstance(mapping, dict):
            raise ValueError("session_map.json requires schema_version=1 and a settings object")
        for relative, setting in mapping.items():
            directory = (root / relative).resolve()
            if (not directory.is_relative_to(root.resolve()) or setting not in labels
                    or not (directory / "manifest.json").is_file()):
                raise ValueError(f"Invalid or missing session mapping: {relative}")
    assignments = []
    included, excluded, signatures = [], [], set()
    sources = [(Path(row["path"])/"manifest.json", row["setting"], row["manifest"]) for row in explicit["entries"]] if explicit is not None else [
        (path, None, None) for path in sorted(root.glob("*/*/*/manifest.json"))]
    for path, report_setting, saved in sources:
        manifest = saved if saved is not None else json.loads(path.read_text(encoding="utf-8"))
        setting = report_setting if explicit is not None else mapping.get(path.parent.relative_to(root).as_posix(), manifest.get("setting"))
        if manifest.get("kind") != "network_characterization":
            if explicit is not None:
                raise ValueError(f"Selected run is not network characterization: {path.parent}")
            continue
        if explicit is None and manifest.get("split") != split:
            continue
        if setting not in settings:
            continue
        if manifest.get("workload", {}).get("workload") != workload:
            if explicit is not None:
                raise ValueError(f"Selected run has the wrong workload: {path.parent}")
            continue
        if manifest.get("status") != "complete":
            if explicit is not None:
                raise ValueError(f"Selected run is not complete ({manifest.get('status')}): {path.parent}")
            excluded.append(dict(path=str(path.parent.resolve()), status=manifest.get("status")))
            continue
        if explicit is not None and not (path.parent/"trace.csv").is_file():
            raise ValueError(f"Selected run is missing trace.csv: {path.parent}")
        signature = {key: manifest.get(key) for key in ("host", "port", "interval_ms", "warmup_requests", "wifi_interval_ms", "wifi_enabled")}
        signature["workload"] = manifest["workload"]
        signatures.add(json.dumps(signature, sort_keys=True))
        included.append(str(path.parent.resolve()))
        assignments.append(dict(path=str(path.parent.resolve()), saved_setting=manifest.get("setting"),
                                saved_split=manifest.get("split"), report_setting=setting))
    if not included:
        raise ValueError(f"No complete {split}/{workload} runs under {root}")
    if len(signatures) != 1:
        raise ValueError("Runs differ in workload, checkpoint, endpoint or cadence. Export comparable runs from a separate experiment group.")
    return dict(included_runs=included, excluded_runs=excluded, split=split, workload=workload,
        comparison_signature=json.loads(next(iter(signatures))), requested_settings=settings,
        missing_settings=[setting for setting in settings if not any(r["report_setting"] == setting for r in assignments)],
        session_map=str(mapping_path.resolve()) if explicit is None and mapping_path.exists() else None,
        session_assignments=assignments, run_selection=selection_audit(explicit) if explicit is not None else None)


def export_table(root, output, split="test", workload="policy", settings=None, selection_config=None):
    """Export only explicitly listed runs when selection_config is supplied."""
    output = Path(output)
    selection = select_network_runs(root, split, workload, settings, selection_config)
    settings = selection["requested_settings"]
    labels = {"strong": "Strong Wi-Fi", "weak": "Weak Wi-Fi", "stationary": "Stationary", "walking": "Walking route"}
    groups = {}
    for assignment in selection["session_assignments"]:
        directory = Path(assignment["path"])
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        data = _measurements(directory, manifest)
        group = groups.setdefault(assignment["report_setting"], [[], [], [], []])
        for dest, src in zip(group, data):
            dest.extend(src)
    table, audits = [], {}
    for setting in settings:
        if setting not in groups:
            continue
        rows, delays, polls, signals = groups[setting]
        table.append(dict(setting=setting, completed_requests=len(delays),
                          median_rssi_dbm=statistics.median(signals) if signals else None,
                          response_p50_ms=quantile(delays, .5) if delays else None,
                          response_p95_ms=quantile(delays, .95) if delays else None))
        audits[setting] = dict(failed_requests=sum(r["status"] != "ok" for r in rows),
                               wifi_polls=len(polls), valid_rssi_polls=len(signals))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "table1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = ReportWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows([{k: "" if v is None else v for k, v in row.items()} for row in table])
    latex = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Setting & Requests & Median RSSI (dBm) & Response p50 (ms) & Response p95 (ms) \\", r"\midrule"]
    for row in table:
        cells = [labels[row["setting"]]]
        cells += [r"\textemdash" if value is None else f"{value:.2f}" for key, value in row.items() if key != "setting"]
        latex.append(" & ".join(cells) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}"]
    (output / "table1.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(dict(**selection, table=table, audits=audits,
        note="Pooled successful requests and independent radio polls within setting; no cross-setting average"), indent=2), encoding="utf-8")
    if selection["run_selection"] is not None:
        (output / "selection.json").write_text(json.dumps(selection["run_selection"], indent=2), encoding="utf-8")
    print(f"Table 1: {output.resolve()}", flush=True)
    for setting, audit in audits.items():
        print(f"{setting}: {audit['failed_requests']} request errors; {audit['valid_rssi_polls']}/{audit['wifi_polls']} RSSI polls", flush=True)
    if selection["excluded_runs"]:
        print(f"Excluded {len(selection['excluded_runs'])} interrupted/failed runs; see summary.json", flush=True)
    return output
