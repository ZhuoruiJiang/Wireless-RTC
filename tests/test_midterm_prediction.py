import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wireless_rtc.analysis.midterm_prediction import evaluate_table2, latest_table1


class MidtermPredictionTests(unittest.TestCase):
    def fixture(self, root):
        signature = dict(workload=dict(workload="policy", task="cartpole_thrust"), host="localhost")
        assignments = []
        for name, values in [("first", [10, 10, 30, 10]), ("second", [10, 10, 10, 10])]:
            directory = root/name
            directory.mkdir()
            (directory/"manifest.json").write_text(json.dumps(dict(**signature,
                kind="network_characterization", status="complete", response_duration_clock="perf_counter_ns")))
            with (directory/"trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["status", "actual_ms", "wall_time_ns"])
                writer.writeheader()
                writer.writerows(dict(status="ok", actual_ms=value, wall_time_ns=i*100_000_000) for i,value in enumerate(values))
            assignments.append(dict(path=str(directory), saved_setting="walking", report_setting="weak"))
        selection = root/"table1.json"
        selection.write_text(json.dumps(dict(session_assignments=assignments, requested_settings=["weak"],
            missing_settings=[], workload="policy", split="test", comparison_signature=signature)))
        return selection

    def test_causal_history_and_reset_and_step_metrics(self):
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"):
            root = Path(temp)
            selection = self.fixture(root)
            before = (root/"first/trace.csv").read_bytes()
            output = evaluate_table2(selection, root/"report", history=2)
            metrics = json.loads((output/"metrics.json").read_text())
            last = next(r for r in metrics if r["predictor"] == "last_response")
            self.assertEqual(last["samples"], 4)
            self.assertEqual(last["mae_ms"], 10)
            self.assertEqual(last["step_mae"], .5)
            self.assertEqual(last["step_underprediction_pct"], 25)
            self.assertEqual(last["step_overprediction_pct"], 25)
            self.assertEqual(last["mean_excess_steps"], .25)
            self.assertEqual(last["coverage_pct"], 75)
            self.assertEqual(last["exceedance_pct"], 25)
            self.assertFalse(any(key.endswith("_rate") or key == "coverage" for key in last))
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual([a["excluded_warmup"] for a in manifest["per_session"]], [2,2])
            self.assertEqual(before, (root/"first/trace.csv").read_bytes())
            self.assertIn("Fugu-inspired MLP", (output/"table2.tex").read_text())
            self.assertEqual(len(metrics), 7)
            with (output/"table2.csv").open(newline="") as stream:
                report = list(csv.DictReader(stream))
            self.assertEqual([r["predictor"] for r in report], ["last_response", "recent_max", "rolling_p95"])
            self.assertNotIn("Rolling p50", (output/"table2.tex").read_text())
            with (output/"internal_metrics.csv").open(newline="") as stream:
                internal = list(csv.DictReader(stream))
            self.assertEqual(len(internal), 7)
            self.assertEqual(float(internal[0]["coverage_pct"]), 75)
            self.assertEqual(float(internal[0]["step_underprediction_pct"]), 25)
            self.assertTrue(all(row["samples"] == 4 for row in metrics))
            with (output/"predictions.csv").open(newline="") as stream:
                predictions = list(csv.DictReader(stream))
            # Past delays at this target are [10,30]; p50 is 20, p70 is 24.
            forecasts = {row["predictor"]: float(row["predicted_ms"]) for row in predictions
                         if Path(row["session"]).name == "first" and row["index"] == "3"}
            self.assertEqual(forecasts["rolling_p50"], 20)
            self.assertAlmostEqual(forecasts["rolling_p70"], 24)
            ordered = [forecasts[name] for name in ["rolling_p50", "rolling_p70", "rolling_p80", "rolling_p90", "rolling_p95", "recent_max"]]
            self.assertEqual(ordered, sorted(ordered))
            under = [row["step_underprediction_pct"] for row in metrics if row["predictor"].startswith("rolling_")]
            self.assertEqual(under, sorted(under, reverse=True))
            with self.assertRaises(FileExistsError):
                evaluate_table2(selection, output, history=2)

    def test_reject_censored_data_instead_of_silently_omitting_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selection = self.fixture(root)
            path = root/"first/trace.csv"
            path.write_text(path.read_text().replace("ok,30", "error,30"))
            with self.assertRaisesRegex(ValueError, "censored"):
                evaluate_table2(selection, root/"report", history=2)
            self.assertFalse((root/"report").exists())

    def test_reject_changed_workload_and_old_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selection = self.fixture(root)
            path = root/"first/manifest.json"
            original = json.loads(path.read_text())
            changed = dict(original, host="other")
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "no longer match"):
                evaluate_table2(selection, root/"report", history=2)
            original["response_duration_clock"] = "monotonic_ns"
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "high-resolution"):
                evaluate_table2(selection, root/"report", history=2)

    def test_latest_complete_table1_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, missing in [("01", []), ("02", ["strong"])]:
                path = root/"reports"/name
                path.mkdir(parents=True)
                (path/"summary.json").write_text(json.dumps(dict(workload="policy", split="test",
                    requested_settings=["strong", "weak"], missing_settings=missing)))
            self.assertEqual(latest_table1(root, ["strong", "weak"]).parent.name, "01")

    def test_explicit_prediction_selection_needs_no_table1_report(self):
        from wireless_rtc.network.characterization import select_network_runs
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"):
            root = Path(temp)
            self.fixture(root)
            config = root/"prediction.json"
            config.write_text(json.dumps(dict(schema_version=1,runs={"weak":["first","second"]})))
            selection = select_network_runs(root,selection_config=config)
            output = evaluate_table2(None,root/"report",history=2,selection=selection)
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertIsNone(manifest["table1"])
            self.assertEqual(len(manifest["inputs"]),2)
            self.assertTrue((output/"selection.json").exists())
