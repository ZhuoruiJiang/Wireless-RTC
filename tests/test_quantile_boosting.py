"""Leakage, holdout, censoring and artifact checks; fixtures are not research data."""
import contextlib
import csv
from dataclasses import replace
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from wireless_rtc.prediction.dataset import Record, export_records, partition_records, read_records
from wireless_rtc.prediction.features import build_examples, feature_names, feature_values
from wireless_rtc.prediction.quantile_boosting import (evaluate_quantiles, load_bundle, score,
                                                       train_quantiles, validate_config)


def config():
    return dict(history=3, quantiles=[.5,.8,.95], feature_sets=["rtt","rtt_snr"],
                split_mode="chronological", train_fraction=.6, validation_fraction=.2,
                max_gap_ms=10000, control_ms=20, seed=42,
                parameter_grid=[dict(max_iter=5,max_leaf_nodes=3,min_samples_leaf=5)])


def records(n=180):
    return [Record("run-a",i*100.,10.+i%7,15.+i%11) for i in range(n)]


class FeatureTests(unittest.TestCase):
    def test_comparison_defaults_to_full_source_with_per_session_warmup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/"source.csv"
            sessions = {"a":records(20),"b":[replace(r,session="run-b") for r in records(20)]}
            export_records(source,sessions)
            script = Path(__file__).resolve().parents[1]/"scripts/compare_rtt_segment.py"
            result = subprocess.run([sys.executable,str(script),"--source",str(source),"--output",str(root/"result")],
                                    capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest = json.loads((root/"result/manifest.json").read_text())
            self.assertEqual(manifest["requested_samples"],40)
            self.assertEqual(manifest["skipped_warmup_samples"],16)
            self.assertEqual(manifest["samples"],24)
            metrics = json.loads((root/"result/metrics.json").read_text())
            self.assertTrue(all(r["samples"]==24 for r in metrics))

    def test_missing_input_explains_placeholder_instead_of_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"prepare a real dataset"):
                read_records([Path(temp)/"missing.csv"])

    def test_current_target_rtt_and_snr_do_not_enter_features(self):
        rows = records(10)
        examples,_ = build_examples({"a":rows},3,10000)
        before = feature_values(examples[0]["history"],"rtt_snr")
        rows[3] = replace(rows[3],rtt_ms=99999.,snr_db=-500.)
        after,_ = build_examples({"a":rows},3,10000)
        self.assertEqual(before,feature_values(after[0]["history"],"rtt_snr"))
        self.assertEqual(after[0]["target"].rtt_ms,99999.)

    def test_partitions_share_no_history_or_target_observations(self):
        sessions = partition_records({"a":records()},config())
        used = []
        for part in ("train","validation","test"):
            examples,_ = build_examples(sessions,3,10000,part)
            used.append({r.identity for e in examples for r in e["history"]+[e["target"]]})
        self.assertTrue(all(used))
        self.assertFalse(used[0]&used[1] or used[0]&used[2] or used[1]&used[2])

    def test_errors_and_gaps_reset_context_and_are_counted(self):
        rows = records(10)
        rows[3] = replace(rows[3],rtt_ms=None,status="timeout")
        examples,audit = build_examples({"a":rows},3,10000)
        self.assertEqual(examples[0]["target"],rows[7])
        self.assertEqual(audit["failed_records"],1)
        rows[3] = replace(records(10)[3],timestamp_ms=100000.)
        _,audit = build_examples({"a":rows[:4]},3,10000)
        self.assertEqual(audit["gap_resets"],1)

    def test_missing_snr_stays_missing_and_age_is_preserved(self):
        rows = [Record("a",0.,10.,20.,snr_timestamp_ms=0.),
                Record("a",100.,12.,20.,snr_timestamp_ms=0.), Record("a",200.,15.)]
        values = feature_values(rows,"rtt_snr")
        mapped = dict(zip(feature_names(3,"rtt_snr"),values))
        self.assertTrue(math.isnan(mapped["snr_lag_1_db"]))
        self.assertEqual(mapped["snr_age_ms"],200.)

    def test_invalid_order_future_snr_and_mixed_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"data.csv"
            for body in ["timestamp_ms,rtt_ms\n1,10\n1,11\n",
                         "timestamp_ms,rtt_ms,snr_db,snr_timestamp_ms\n1,10,20,2\n"]:
                path.write_text(body)
                with self.assertRaises(ValueError):
                    read_records([path])
        with self.assertRaises(ValueError):
            partition_records({"a":[replace(records()[0],split="train"),records()[1]]},config())

    def test_configuration_and_metrics(self):
        c = config()
        for update in [dict(quantiles=[.95,.5]),dict(history=1),dict(train_fraction=.9),
                       dict(feature_sets=["bad"]),dict(parameter_grid=[dict(early_stopping=True)])]:
            with self.assertRaises(ValueError):
                validate_config(dict(c,**update))
        metrics = score([10.,40.],[20.,20.],.8,20.)
        self.assertEqual(metrics["mae_ms"],15.)
        self.assertEqual(metrics["coverage"],.5)
        self.assertAlmostEqual(metrics["pinball_loss_ms"],9.)
        self.assertEqual(metrics["step_underprediction_rate"],.5)


@unittest.skipUnless(importlib.util.find_spec("sklearn"),"Install the prediction extra for learned model checks")
class TrainingTests(unittest.TestCase):
    def test_segment_comparison_with_optional_qgb_and_no_snr_columns(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            c = dict(config(),feature_sets=["rtt"])
            sessions = partition_records({"a":records()},c)
            source, segment = root/"source.csv", root/"segment.csv"
            export_records(source,sessions)
            targets = [r for r in sessions["a"] if r.split=="test"][c["history"]:]
            with segment.open("w",newline="") as stream:
                writer = csv.DictWriter(stream,fieldnames=["session_id","timestamp_ms","rtt_ms","status","split"])
                writer.writeheader()
                writer.writerows(dict(session_id=r.session,timestamp_ms=r.timestamp_ms,rtt_ms=r.rtt_ms,
                                      status=r.status,split="test") for r in targets)
            train_quantiles([source],c,root/"train")
            script = Path(__file__).resolve().parents[1]/"scripts/compare_rtt_segment.py"
            command = [sys.executable,str(script),"--source",str(source),"--segment",str(segment),
                       "--history",str(c["history"]),"--model",str(root/"train/model.pkl"),"--output",str(root/"evaluation")]
            result = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            metrics = json.loads((root/"evaluation/metrics.json").read_text())
            self.assertEqual({r["model"] for r in metrics},
                             {"last_value","rolling_mean","rolling_median","recent_max","qgb_rtt"})
            self.assertTrue(all(r["samples"]==len(targets) for r in metrics))

    def test_cli_training_and_heldout_evaluation(self):
        from wireless_rtc.cli import main
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            data, settings = root/"data.csv", root/"config.json"
            export_records(data,{"a":records()})
            settings.write_text(json.dumps(config()))
            main(["quantile-train","--data",str(data),"--config",str(settings),"--output",str(root/"train")])
            main(["quantile-evaluate","--data",str(data),"--model",str(root/"train/model.pkl"),
                  "--output",str(root/"test")])
            self.assertEqual(json.loads((root/"test/manifest.json").read_text())["kind"],"frozen_quantile_model_test")

    def test_train_save_load_frozen_test_and_overlap_guard(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = root/"measured.csv"
            export_records(data,{"a":records()})
            train_quantiles([data],config(),root/"train")
            bundle = load_bundle(root/"train/model.pkl")
            preds = bundle.predict(records()[-3:])
            self.assertEqual(list(preds),[.5,.8,.95])
            self.assertEqual(list(preds.values()),sorted(preds.values()))
            evaluate_quantiles(root/"train/model.pkl",[data],root/"test")
            manifest = json.loads((root/"train/manifest.json").read_text())
            self.assertEqual(set(manifest["audits"]),{"train","validation"})
            self.assertEqual(manifest["evaluation_partition"],"validation")
            self.assertTrue((root/"test/predictions.csv").exists())
            with (root/"test/metrics.csv").open() as stream:
                report = list(csv.DictReader(stream))
            self.assertEqual({r["model"] for r in report},
                             {"last_value","rolling_median","recent_max","rolling_quantile","qgb_rtt","qgb_rtt_snr"})
            for row in report:
                self.assertRegex(row["mae_ms"],r"^\d+\.\d{2}$")
            with self.assertRaisesRegex(ValueError,"overlap"):
                evaluate_quantiles(root/"train/model.pkl",[data],root/"bad",partition="all")
            with self.assertRaises(FileExistsError):
                train_quantiles([data],config(),root/"train")

    def test_test_targets_do_not_affect_training_or_selection(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            original = partition_records({"a":records()},config())
            changed = {k:[replace(r,rtt_ms=9000.,snr_db=-300.) if r.split=="test" else r for r in rows]
                       for k,rows in original.items()}
            for name, data in [("original",original),("changed",changed)]:
                path = root/f"{name}.csv"
                export_records(path,data)
                train_quantiles([path],config(),root/name)
            a = load_bundle(root/"original/model.pkl")
            b = load_bundle(root/"changed/model.pkl")
            self.assertEqual(a.predict(records()[:3]),b.predict(records()[:3]))
            ma = json.loads((root/"original/manifest.json").read_text())
            mb = json.loads((root/"changed/manifest.json").read_text())
            self.assertEqual(ma["selected_candidates"],mb["selected_candidates"])

    def test_missing_snr_requires_explicit_rtt_only_experiment(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(temp)/"data.csv"
            export_records(path,{"a":[replace(r,snr_db=None) for r in records()]})
            with self.assertRaisesRegex(ValueError,"No historical SNR"):
                train_quantiles([path],config(),Path(temp)/"model")
