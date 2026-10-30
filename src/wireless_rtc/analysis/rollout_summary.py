"""Tables 3/4 and internal prediction-success diagnostics from live rollout blocks."""
import csv
import json
from pathlib import Path
import random
import statistics

from wireless_rtc.analysis.formatting import ReportWriter
from wireless_rtc.client.schedule import delay_steps
from wireless_rtc.control_modes import condition_name
from wireless_rtc.prediction.rolling import quantile


MAIN = [("naive", "none"), ("rtc", "recent_max"), ("wireless_rtc", "last_response"),
        ("wireless_rtc", "recent_max"), ("wireless_rtc", "rolling_p95")]
BEHAVIOR = [("rtc", "recent_max"), ("wireless_rtc", "recent_max"), ("wireless_rtc", "rolling_p95")]


def _read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _true(value):
    return str(value).lower() in {"true", "1"}


def _percent(rows, field):
    return 100 * sum(_true(r[field]) for r in rows)/len(rows) if rows else None


def _macro(rows, field):
    return statistics.mean(r[field] for r in rows) if all(r[field] is not None for r in rows) else None


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = ReportWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{key: "" if value is None else value for key,value in row.items()} for row in rows])


def _task_metrics(episodes, requests, ticks, control_ms):
    eligible = [r for r in requests if _true(r.get("deadline_evaluable"))]
    consumed = [r for r in requests if r["status"] == "ok" and r["arrival_tick"] != ""]
    jumps = [float(r["boundary_action_jump"]) for r in ticks if r.get("boundary_action_jump", "") != ""]
    forecast, seen = [], {}
    for row in consumed:
        episode = row["episode"]
        count = seen.get(episode, 0)
        seen[episode] = count+1
        if count >= 8 and row["predicted_ms"] != "":
            predicted, actual = float(row["predicted_ms"]), float(row["actual_ms"])
            forecast.append((delay_steps(actual, control_ms), delay_steps(predicted, control_ms)))
    return dict(episodes=len(episodes), successes=sum(_true(r["success"]) for r in episodes),
        success_pct=100*sum(_true(r["success"]) for r in episodes)/len(episodes),
        deadline_miss_pct=_percent(eligible, "deadline_missed"),
        expired_response_pct=_percent(consumed, "stale_response"),
        fallback_tick_pct=_percent(ticks, "buffer_exhausted"),
        boundary_jump_p95=quantile(jumps, .95) if jumps else None,
        deadline_requests=len(eligible), deadline_censored_requests=len(requests)-len(eligible),
        consumed_responses=len(consumed), control_ticks=len(ticks), valid_boundaries=len(jumps),
        prediction_samples=len(forecast),
        step_underprediction_pct=100*sum(a>p for a,p in forecast)/len(forecast) if forecast else None,
        step_overprediction_pct=100*sum(p>a for a,p in forecast)/len(forecast) if forecast else None,
        mean_excess_steps=statistics.mean(max(0,p-a) for a,p in forecast) if forecast else None,
        mean_prefix_steps=statistics.mean(p for a,p in forecast) if forecast else None)


def export_rollouts(root, output, settings=None, allow_partial=False, selection_config=None):
    root, output = Path(root), Path(output)
    from wireless_rtc.analysis.run_selection import load_run_selection, selection_audit
    explicit = load_run_selection(selection_config, settings) if selection_config is not None else None
    if explicit is not None:
        settings = explicit["settings"]
    elif settings is None:
        settings = ("strong", "weak")
    runs, excluded, signature = {}, [], None
    sources = [(Path(row["path"])/"manifest.json", row["setting"], row["manifest"]) for row in explicit["entries"]] if explicit is not None else [
        (path, None, None) for path in sorted(root.glob("*/*/*/manifest.json"))]
    for path, report_setting, saved in sources:
        m = saved if saved is not None else json.loads(path.read_text(encoding="utf-8"))
        if explicit is None and (m.get("split") != "test" or m.get("setting") not in settings):
            continue
        if m.get("status") != "completed":
            if explicit is not None:
                raise ValueError(f"Selected rollout is not completed ({m.get('status')}): {path.parent}")
            excluded.append(dict(path=str(path.parent.resolve()), status=m.get("status")))
            continue
        if m.get("rollout_metrics_version") != 1:
            raise ValueError(f"Run predates Table 4 logging: {path}")
        config = {k:v for k,v in m["config"].items() if k != "block"}
        key = json.dumps(dict(config=config, policies={t: m["tasks"][t]["server"].get("checkpoint_sha256")
                                                       for t in m["selected_tasks"]}), sort_keys=True)
        if signature is not None and signature != key:
            raise ValueError(f"Selected rollout configurations/checkpoints or task sets differ at {path.parent}; select matching sweeps")
        signature = key
        setting, block = report_setting if explicit is not None else m["setting"], m["block"]
        if (setting, block) in runs:
            raise ValueError(f"Duplicate block {block} for {setting}; select one attempt rather than counting seeds twice")
        runs[setting, block] = (path.parent, m)
    if not runs:
        raise ValueError("No completed test rollout blocks found")
    reference = next(iter(runs.values()))[1]
    tasks = reference["selected_tasks"]
    conditions = [(c["method"],c["predictor"]) for c in reference["conditions"]]
    expected_blocks = reference["blocks"]
    missing = {setting: [b for b in range(1, expected_blocks+1) if (setting,b) not in runs] for setting in settings}
    if any(missing.values()) and not allow_partial:
        raise ValueError(f"Missing blocks {missing}; finish them or use --allow-partial for a preliminary export")
    buckets = {}
    block_success = {}
    for (setting, block), (directory, m) in runs.items():
        if m["selected_tasks"] != tasks or [(c["method"],c["predictor"]) for c in m["conditions"]] != conditions:
            raise ValueError("Different task/condition sweeps")
        for task in tasks:
            for method, predictor in conditions:
                folder = directory/"tasks"/task/condition_name(method,predictor)
                episodes, requests, ticks = [_read(folder/name) for name in ("episodes.csv", "requests.csv", "ticks.csv")]
                expected = set(range(m["episode_offset"], m["episode_offset"]+m["episodes_per_task_condition"]))
                if (len(episodes) != len(expected) or {int(r["episode"]) for r in episodes} != expected
                        or any(r["status"] != "ok" for r in episodes)):
                    raise ValueError(f"Incomplete/error episode set: {folder}")
                if {int(r["seed"]) for r in episodes} != {(m["config"]["seed"]+e) % 2**32 for e in expected}:
                    raise ValueError(f"Episode seed schedule differs: {folder}")
                bucket = buckets.setdefault((setting,task,method,predictor), [[], [], []])
                for dest, source in zip(bucket, (episodes, requests, ticks)):
                    dest.extend(source)
                block_success[setting,block,task,method,predictor] = statistics.mean(_true(r["success"]) for r in episodes)
    per_task, overall = [], []
    for setting in settings:
        blocks = sorted(b for s,b in runs if s == setting)
        if not blocks:
            continue
        for method,predictor in conditions:
            rows = []
            for task in tasks:
                result = _task_metrics(*buckets[setting,task,method,predictor], reference["config"]["control_ms"])
                row = dict(setting=setting,task=task,method=method,predictor=predictor,**result)
                rows.append(row)
                per_task.append(row)
            aggregate = {name: _macro(rows,name) for name in ("success_pct", "deadline_miss_pct", "expired_response_pct",
                "fallback_tick_pct", "boundary_jump_p95", "step_underprediction_pct", "step_overprediction_pct",
                "mean_excess_steps", "mean_prefix_steps")}
            low = high = None
            if len(blocks) >= 2:
                # Resample complete blocks jointly across tasks, rather than
                # pretending correlated requests are independent task trials.
                rng = random.Random(42)
                block_means = [statistics.mean(block_success[setting,b,t,method,predictor] for t in tasks) for b in blocks]
                bootstrap = [100*statistics.mean(rng.choices(block_means,k=len(blocks))) for _ in range(4000)]
                low, high = quantile(bootstrap,.025), quantile(bootstrap,.975)
            overall.append(dict(setting=setting,method=method,predictor=predictor,blocks=len(blocks),
                episodes_per_task=rows[0]["episodes"], success_ci_low_pct=low, success_ci_high_pct=high, **aggregate))
    output.mkdir(parents=True,exist_ok=False)
    table3 = [r for r in overall if (r["method"],r["predictor"]) in MAIN]
    table4 = [{k:r[k] for k in ("setting", "method", "predictor", "deadline_miss_pct", "expired_response_pct", "fallback_tick_pct", "boundary_jump_p95")}
              for r in overall if (r["method"],r["predictor"]) in BEHAVIOR]
    if not table3 or not table4:
        raise ValueError("Main report conditions missing; use the midterm rollout config")
    _write(output/"table3.csv", [{k:r[k] for k in ("setting", "method", "predictor", "blocks", "episodes_per_task", "success_pct", "success_ci_low_pct", "success_ci_high_pct")} for r in table3])
    _write(output/"table4.csv", table4)
    _write(output/"internal_rollout_metrics.csv", overall)
    _write(output/"per_task_metrics.csv", per_task)
    _write(output/"tradeoff.csv", [r for r in overall if r["method"] == "wireless_rtc"])
    def cell(value):
        return r"\textemdash" if value is None else f"{value:.2f}"
    indexed = {(r["setting"],r["method"],r["predictor"]):r for r in overall}
    latex3 = [r"\begin{tabular}{ll" + "r"*len(settings) + "}", r"\toprule",
        "Controller & Predictor & " + " & ".join(s.title()+r" (\%)" for s in settings)+r" \\", r"\midrule"]
    for method,predictor in MAIN:
        values = []
        for setting in settings:
            row = indexed.get((setting,method,predictor))
            values.append(cell(row["success_pct"]) + (" ["+cell(row["success_ci_low_pct"])+", "+cell(row["success_ci_high_pct"])+"]" if row["success_ci_low_pct"] is not None else "") if row else cell(None))
        latex3.append(method.replace("_",r"\_")+" & "+predictor.replace("_",r"\_")+" & "+" & ".join(values)+r" \\")
    latex3 += [r"Wireless RTC & MLP & \multicolumn{"+str(len(settings))+r"}{c}{\textit{Pending}} \\",r"\bottomrule",r"\end{tabular}"]
    latex4 = [r"\begin{tabular}{llrrrr}",r"\toprule",r"Setting & Method & Misses (\%) & Expired (\%) & Fallback (\%) & Jump \\",r"\midrule"]
    for row in table4:
        latex4.append(row["setting"]+" & "+condition_name(row["method"],row["predictor"]).replace("_",r"\_")+" & "+" & ".join(cell(row[k]) for k in ("deadline_miss_pct","expired_response_pct","fallback_tick_pct","boundary_jump_p95"))+r" \\")
    latex4 += [r"\bottomrule",r"\end{tabular}"]
    (output/"table3.tex").write_text("\n".join(latex3)+"\n",encoding="utf-8")
    (output/"table4.tex").write_text("\n".join(latex4)+"\n",encoding="utf-8")
    report = dict(overall=overall,per_task=per_task,missing_blocks=missing,excluded_runs=excluded,
        selected_tasks=tasks, run_selection=selection_audit(explicit) if explicit is not None else None,
        included_runs=[str(p.resolve()) for p,_ in runs.values()],
        uncertainty="95% percentile bootstrap of complete blocks, jointly across tasks; preliminary with few blocks; unavailable for one block",
        prediction_scoring="Consumed successful responses only, after eight per-episode warmup responses; unbounded predicted steps vs full RTT steps",
        aggregation="Equal-weight mean of per-task metrics. Boundary jump is mean of per-task p95 values; unavailable if any task lacks boundaries.")
    (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if explicit is not None:
        (output/"selection.json").write_text(json.dumps(selection_audit(explicit),indent=2),encoding="utf-8")
    print(f"Tables 3/4: {output.resolve()}",flush=True)
    return output
