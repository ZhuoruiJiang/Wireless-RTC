from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wireless_rtc.cli import main
from wireless_rtc.run_paths import resolve_output


class RunPathTests(unittest.TestCase):
    def test_experiment_setting_split_and_legacy_paths(self):
        args = SimpleNamespace(output=None, tag=None, command="network-characterize", output_root="runs",
                               experiment="midterm", setting="walking", split="test")
        result = resolve_output(args)
        self.assertEqual(result.parent, Path("runs/midterm/network-characterization/walking/test"))
        args.experiment = "../escape"
        with self.assertRaises(ValueError):
            resolve_output(args)
        args.experiment = None
        with self.assertRaises(ValueError):
            resolve_output(args)

    def test_tag_datetime_and_collision_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output=None,tag="campus",command="rtc-live",output_root=temp)
            now = datetime(2026,10,27,12,34,56,123456,tzinfo=timezone.utc)
            first = resolve_output(args,now)
            self.assertEqual(first.name,"campus-20261027-123456-123456Z")
            first.mkdir()
            second = resolve_output(args,now)
            self.assertEqual(second.name,first.name+"-02")
            self.assertTrue(first.is_dir())

    def test_default_tag_exact_path_and_invalid_tag(self):
        args = SimpleNamespace(output=None,tag=None,command="rtc-live",output_root="runs")
        self.assertTrue(resolve_output(args).name.startswith("rtc-live-"))
        args.output = "runs/exact"
        self.assertEqual(resolve_output(args),Path("runs/exact"))
        args.output,args.tag = None,"../escape"
        with self.assertRaises(ValueError):
            resolve_output(args)

    def test_repeated_cli_command_creates_separate_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp)/"config.json"
            config.write_text(json.dumps(dict(seed=42,samples=3,warmup=0,control_ms=20,scoring_quantile=.95,
                                              sources={"fixed":{"kind":"fixed","value_ms":10}},
                                              predictors={"last":{"kind":"last"}})))
            args = ["prediction-suite","--config",str(config),"--tag","repeat","--output-root",str(Path(temp)/"runs")]
            with patch("sys.stdout"):
                main(args)
                main(args)
            runs = list((Path(temp)/"runs").iterdir())
            self.assertEqual(len(runs),2)
            self.assertTrue(all((p/"comparison.csv").is_file() for p in runs))
