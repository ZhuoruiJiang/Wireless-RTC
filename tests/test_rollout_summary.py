import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wireless_rtc.analysis.rollout_summary import export_rollouts, MAIN
from wireless_rtc.client.live import validate_config
from wireless_rtc.control_modes import condition_name


def write_csv(path, rows):
    with path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


class RolloutSummaryTests(unittest.TestCase):
    def make_run(self, root, setting, block):
        config = dict(seed=42,episodes=4,blocks=2,block=block,control_ms=20)
        tasks = ["cartpole_thrust", "catcher_v3", "hard_lunar_lander"]
        conditions = MAIN+[("wireless_rtc",f"rolling_p{q}") for q in (50,70,80,90)]
        directory = root/setting/"test"/f"block{block}"
        directory.mkdir(parents=True)
        metadata = dict(split="test",setting=setting,status="completed",config=config,block=block,blocks=2,
            rollout_metrics_version=1,selected_tasks=tasks,episode_offset=(block-1)*2,
            episodes_per_task_condition=2,conditions=[dict(method=m,predictor=p) for m,p in conditions],
            tasks={task:dict(server=dict(checkpoint_sha256=task)) for task in tasks})
        (directory/"manifest.json").write_text(json.dumps(metadata))
        for task in tasks:
            for method,predictor in conditions:
                folder = directory/"tasks"/task/condition_name(method,predictor)
                folder.mkdir(parents=True)
                episodes, requests, ticks = [], [], []
                for episode in range((block-1)*2,block*2):
                    episodes.append(dict(episode=episode,seed=42+episode,status="ok",success=(task==tasks[0] or block==2)))
                    for i in range(10):
                        requests.append(dict(episode=episode,status="ok",arrival_tick=i+1,deadline_evaluable=True,
                            deadline_missed=i>=5,stale_response=i==9,actual_ms=20,
                            predicted_ms="" if method=="naive" else 0 if predictor=="rolling_p50" else 40))
                    ticks += [dict(episode=episode,buffer_exhausted=False,boundary_action_jump=""),
                              dict(episode=episode,buffer_exhausted=False,boundary_action_jump=.2),
                              dict(episode=episode,buffer_exhausted=True,boundary_action_jump="")]
                for filename,rows in [("episodes.csv",episodes),("requests.csv",requests),("ticks.csv",ticks)]:
                    write_csv(folder/filename,rows)
        return directory

    def test_tables_internal_metrics_and_macro_aggregation(self):
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"):
            root = Path(temp)/"runs"
            for setting in ("strong","weak"):
                for block in (1,2):
                    self.make_run(root,setting,block)
            output = export_rollouts(root,Path(temp)/"report")
            summary = json.loads((output/"summary.json").read_text())
            self.assertEqual(len(summary["overall"]),18)
            row = next(r for r in summary["overall"] if r["setting"]=="strong" and r["predictor"]=="recent_max")
            self.assertAlmostEqual(row["success_pct"],100*2/3)
            self.assertEqual(row["deadline_miss_pct"],50)
            self.assertEqual(row["expired_response_pct"],10)
            self.assertAlmostEqual(row["fallback_tick_pct"],100/3)
            self.assertEqual(row["boundary_jump_p95"],.2)
            self.assertEqual(row["step_overprediction_pct"],100)
            self.assertIsNotNone(row["success_ci_low_pct"])
            self.assertNotIn("rolling_p50",(output/"table3.csv").read_text())
            self.assertIn("rolling_p50",(output/"tradeoff.csv").read_text())

    def test_partial_exports_are_explicit_and_duplicate_blocks_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"):
            root = Path(temp)/"runs"
            source = self.make_run(root,"strong",1)
            with self.assertRaisesRegex(ValueError,"Missing blocks"):
                export_rollouts(root,Path(temp)/"report")
            output = export_rollouts(root,Path(temp)/"partial",allow_partial=True)
            summary = json.loads((output/"summary.json").read_text())
            self.assertIsNone(summary["overall"][0]["success_ci_low_pct"])
            duplicate = root/"strong/test/duplicate"
            duplicate.mkdir()
            (duplicate/"manifest.json").write_bytes((source/"manifest.json").read_bytes())
            with self.assertRaisesRegex(ValueError,"Duplicate block"):
                export_rollouts(root,Path(temp)/"duplicate-report",allow_partial=True)

    def test_midterm_config_requires_block_and_has_nine_conditions(self):
        config = json.loads(Path("configs/midterm_rollout.json").read_text())
        with self.assertRaisesRegex(ValueError,"Choose --block"):
            validate_config(config)
        config["block"] = 1
        validate_config(config)
        from wireless_rtc.control_modes import predictors_for
        self.assertEqual(sum(len(list(predictors_for(m,config))) for m in config["methods"]),9)
        self.assertEqual(config["tasks"],["catcher_v3","hard_lunar_lander","mjc_walker"])

    def test_explicit_rollout_config_selects_only_named_attempts(self):
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"):
            root = Path(temp)/"runs"
            groups = {}
            for setting in ("strong","weak"):
                groups[setting] = []
                for block in (1,2):
                    directory = self.make_run(root,setting,block)
                    groups[setting].append(directory.relative_to(root).as_posix())
            extra = self.make_run(root,"strong",3)
            m = json.loads((extra/"manifest.json").read_text())
            m["config"]["seed"] = 999
            (extra/"manifest.json").write_text(json.dumps(m))
            cfg = root/"selection.json"
            cfg.write_text(json.dumps(dict(schema_version=1,runs=groups)))
            source = root/groups["weak"][0]/"manifest.json"
            before = source.read_bytes()
            output = export_rollouts(root,Path(temp)/"report",selection_config=cfg)
            report = json.loads((output/"summary.json").read_text())
            self.assertEqual(len(report["included_runs"]),4)
            self.assertEqual(source.read_bytes(),before)
            self.assertTrue((output/"selection.json").exists())
            m = json.loads(source.read_text())
            m["tasks"]["cartpole_thrust"]["server"]["checkpoint_sha256"] = "changed"
            source.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"checkpoints"):
                export_rollouts(root,Path(temp)/"other",selection_config=cfg)

    def test_explicit_rollout_rejects_running_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"runs"
            directory = self.make_run(root,"weak",1)
            path = directory/"manifest.json"
            m = json.loads(path.read_text())
            m["status"] = "running"
            path.write_text(json.dumps(m))
            cfg = root/"selection.json"
            cfg.write_text(json.dumps(dict(schema_version=1,runs={"weak":[directory.relative_to(root).as_posix()]})))
            with self.assertRaisesRegex(ValueError,"not completed"):
                export_rollouts(root,Path(temp)/"report",selection_config=cfg,allow_partial=True)
            self.assertFalse((Path(temp)/"report").exists())
