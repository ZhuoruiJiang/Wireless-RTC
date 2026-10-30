"""Table 2 on the exact sessions selected by a Table 1 export; no fitting."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from wireless_rtc.analysis.formatting import ReportWriter
from wireless_rtc.client.schedule import delay_steps
from wireless_rtc.prediction.dataset import Record
from wireless_rtc.prediction.features import feature_values
from wireless_rtc.prediction.quantile_boosting import load_bundle, learning_dependencies, rearrange, score
from wireless_rtc.prediction.rolling import quantile


ROLLING_QUANTILES = {f"rolling_p{level}": level / 100 for level in (50, 70, 80, 90, 95)}
METHODS = ("last_response", "recent_max", *ROLLING_QUANTILES)
REPORT_METHODS = ("last_response", "recent_max", "rolling_p95")
LABELS = {"last_response": "Last response", "recent_max": "Recent maximum",
          **{name: "Rolling " + name.split("_")[1] for name in ROLLING_QUANTILES}}
COLUMNS = ["setting", "predictor", "mae_ms", "step_mae", "step_underprediction_pct",
           "step_overprediction_pct", "mean_excess_steps", "mean_prefix_steps"]


def latest_table1(root, settings):
    for path in sorted(Path(root).glob("reports/*/summary.json"), reverse=True):
        document = json.loads(path.read_text(encoding="utf-8"))
        if (document.get("workload") == "policy" and document.get("split") == "test"
                and document.get("requested_settings") == list(settings) and not document.get("missing_settings")):
            return path
    raise ValueError("No complete Table 1 export for these settings. Run network-summary --settings strong weak first.")


def evaluate_table2(table1, output, *, history=8, control_ms=20, model=None, selection=None):
    if type(history) is not int or history < 2 or not math.isfinite(control_ms) or control_ms <= 0:
        raise ValueError("Require history >= 2 and a positive finite control interval")
    table1, output = Path(table1) if table1 is not None else None, Path(output)
    if selection is None:
        if table1 is None:
            raise ValueError("Provide an explicit run selection or Table 1 summary")
        selection = json.loads(table1.read_text(encoding="utf-8"))
    assignments = selection.get("session_assignments", [])
    settings = selection.get("requested_settings", [])
    if not assignments or selection.get("missing_settings") or not settings:
        raise ValueError("Table 1 must have complete settings and explicit session assignments")
    if selection.get("workload") != "policy" or selection.get("split") != "test":
        raise ValueError("Expected a policy-workload test Table 1 export")
    if len({row['path'] for row in assignments}) != len(assignments):
        raise ValueError("Duplicate sessions in Table 1 selection")
    bundle = load_bundle(model) if model is not None else None
    if bundle and (bundle.config["history"] != history or .95 not in bundle.config["quantiles"] or "rtt" not in bundle.estimators):
        raise ValueError("Frozen QGB must support RTT-only p95 and the selected history length")
    max_gap_ms = bundle.config["max_gap_ms"] if bundle else 10000
    examples, provenance, audits = {s: [] for s in settings}, [], []
    for assignment in assignments:
        directory = Path(assignment["path"])
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        setting = assignment["report_setting"]
        if setting not in examples or manifest.get("status") != "complete":
            raise ValueError(f"Invalid setting or incomplete session: {directory}")
        expected = selection["comparison_signature"]
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Session settings no longer match Table 1: {directory}")
        if manifest.get("response_duration_clock") != "perf_counter_ns":
            raise ValueError(f"Session predates high-resolution response timing: {directory}")
        path = directory / "trace.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            raw = list(csv.DictReader(stream))
        records = []
        for row in raw:
            if row["status"] != "ok":
                raise ValueError(f"Session has failed/censored requests; resolve scoring protocol explicitly: {directory}")
            timestamp, actual = int(row["wall_time_ns"]) / 1e6, float(row["actual_ms"])
            if not math.isfinite(actual) or actual < 0 or (records and timestamp <= records[-1].timestamp_ms):
                raise ValueError(f"Invalid delay or non-increasing timestamps: {directory}")
            record = Record(session=str(directory.resolve()), timestamp_ms=timestamp, rtt_ms=actual, split="test")
            if bundle and record.identity in bundle.used_record_ids:
                raise ValueError("Frozen model training/validation overlap detected")
            records.append(record)
        if len(records) <= history:
            raise ValueError(f"Not enough responses after warmup: {directory}")
        if any(b.timestamp_ms-a.timestamp_ms > max_gap_ms for a,b in zip(records, records[1:])):
            raise ValueError(f"Long gap crosses history; define a reset protocol before scoring: {directory}")
        for index in range(history, len(records)):
            examples[setting].append((records[index], records[index-history:index], index))
        provenance.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), setting=setting))
        audits.append(dict(session=str(directory.resolve()), setting=setting, responses=len(records),
                           excluded_warmup=history, scored=len(records)-history))
    metrics, predictions = [], []
    for setting, selected in examples.items():
        if not selected:
            raise ValueError(f"No scored requests for {setting}")
        values = {name: [] for name in METHODS}
        for target, context, index in selected:
            delays = [r.rtt_ms for r in context]
            values["last_response"].append(delays[-1])
            values["recent_max"].append(max(delays))
            for name, level in ROLLING_QUANTILES.items():
                values[name].append(quantile(delays, level))
        if bundle:
            np, _, _, limit = learning_dependencies()
            matrix = np.asarray([feature_values(context, "rtt") for _, context, _ in selected], dtype=float)
            with limit(limits=1):
                forecast = rearrange([estimator.predict(matrix) for estimator in bundle.estimators["rtt"]])
            values["qgb_rtt_p95"] = list(map(float, forecast[:, bundle.config["quantiles"].index(.95)]))
        actual = [target.rtt_ms for target, _, _ in selected]
        for method, predicted in values.items():
            if any(not math.isfinite(p) or p < 0 for p in predicted):
                raise ValueError(f"Invalid forecast from {method}")
            result = score(actual, predicted, .95, control_ms)
            result["step_mae"] = statistics.mean(abs(delay_steps(a, control_ms)-delay_steps(p, control_ms))
                                                   for a,p in zip(actual, predicted))
            for key in ("coverage", "exceedance_rate", "step_underprediction_rate", "step_overprediction_rate"):
                result[key.removesuffix("_rate") + "_pct"] = 100 * result.pop(key)
            metrics.append(dict(setting=setting, predictor=method, **result))
            for (target, _, index), p in zip(selected, predicted):
                predictions.append(dict(setting=setting, predictor=method, session=target.session, index=index,
                    timestamp_ms=target.timestamp_ms, actual_ms=target.rtt_ms, predicted_ms=p,
                    actual_steps=delay_steps(target.rtt_ms, control_ms), predicted_steps=delay_steps(p, control_ms)))
    output.mkdir(parents=True, exist_ok=False)
    report = [{key: row[key] for key in COLUMNS} for row in metrics if row["predictor"] in REPORT_METHODS]
    for filename, rows, fields, writer_class in [("table2.csv", report, COLUMNS, ReportWriter),
            ("internal_metrics.csv", metrics, list(metrics[0]), ReportWriter),
            ("predictions.csv", predictions, list(predictions[0]), csv.DictWriter)]:
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = writer_class(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    latex = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Predictor & MAE (ms) & Step MAE & Under (\%) & Over (\%) & Excess & Prefix \\", r"\midrule"]
    for setting in settings:
        label = {"strong": "Strong Wi-Fi", "weak": "Weak Wi-Fi"}.get(setting, setting.replace("_", " "))
        latex.append(r"\multicolumn{7}{l}{\textbf{" + label + r"}} \\")
        for row in report:
            if row["setting"] == setting:
                cells = [LABELS[row["predictor"]]] + [f"{row[key]:.2f}" for key in COLUMNS[2:]]
                latex.append(" & ".join(cells) + r" \\")
        latex += [r"Fugu-inspired MLP & \multicolumn{6}{c}{\textit{Pending}} \\", r"\midrule"]
    latex[-1] = r"\bottomrule"
    latex.append(r"\end{tabular}")
    (output / "table2.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    manifest = dict(table1=str(table1.resolve()) if table1 is not None else None,
        table1_sha256=hashlib.sha256(table1.read_bytes()).hexdigest() if table1 is not None else None,
        run_selection=selection.get("run_selection"),
        history=history, control_ms=control_ms, scoring_quantile=.95,
        rolling_quantiles=ROLLING_QUANTILES, report_methods=REPORT_METHODS,
        rate_units="percent (0-100), columns suffixed _pct", per_session=audits, inputs=provenance,
        model=str(Path(model).resolve()) if model else None,
        model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest() if model else None,
        protocol="Predict before observing; reset at each session; identical target rows for every predictor; no fitting")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if selection.get("run_selection") is not None:
        (output / "selection.json").write_text(json.dumps(selection["run_selection"], indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Table 2: {output.resolve()}", flush=True)
    print(f"Scored {sum(a['scored'] for a in audits)} responses across {len(audits)} sessions; "
          f"excluded {sum(a['excluded_warmup'] for a in audits)} warmup responses.", flush=True)
    return output
