"""Task selection, policy routing and equal-weight scoring without model dependencies."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wireless_rtc.analysis.tasks import aggregate_tasks
from wireless_rtc.client.live import run_remote
from wireless_rtc.control_modes import CONTROL_MODES,PROTOCOL_VERSION
from wireless_rtc.server.policies import PolicyRegistry
from wireless_rtc.server.service import make_server
from wireless_rtc.tasks import TASKS,selected_tasks,task_checkpoint,level_sha256


class Array:
    def __init__(self,values):
        self.values = values
    def tolist(self):
        return self.values


class Simulator:
    def __init__(self,level,seed):
        self.task = Path(level).stem
        self.level_sha256 = hashlib.sha256(level.encode()).hexdigest()
        self.action_dim = 1 if self.task == "grasp_easy" else 2
        self.observation_dim = 1
        self.params = SimpleNamespace(max_timesteps=3)
        random = SimpleNamespace(key=lambda x:x,fold_in=lambda x,y:x+y,split=lambda x:(x+1,x+2))
        self.jax = SimpleNamespace(random=random,devices=lambda:["test-cpu"],block_until_ready=lambda x:x)
        self.jnp = SimpleNamespace(zeros=lambda n:[0.]*n,asarray=lambda x:x)
        self.np = SimpleNamespace(asarray=lambda x:x if isinstance(x,Array) else Array(x))

    def reset(self,key):
        return Array([0.]),0

    def step(self,key,state,action):
        if len(action) != self.action_dim:
            raise ValueError("Action from wrong task policy")
        return Array([state+1]),state+1,0.,state+1 >= 3,{"success":self.task == "grasp_easy"}


class Policy:
    def __init__(self,checkpoint,level):
        self.level,self.task = level,Path(level).stem
        self.action_dim = 1 if self.task == "grasp_easy" else 2
        if Path(checkpoint).name != f"worlds_l_{self.task}.pkl":
            raise ValueError("Wrong checkpoint")

    def describe(self):
        return dict(protocol_version=PROTOCOL_VERSION,control_modes=CONTROL_MODES.copy(),
                    task=self.task,available_tasks=[self.task],level=self.level,horizon=8,
                    observation_dim=1,action_dim=self.action_dim,
                    level_sha256=hashlib.sha256(self.level.encode()).hexdigest(),
                    checkpoint_sha256=f"hash-{self.task}")

    def initialize(self,request):
        return dict(actions=[[0.]*self.action_dim]*8,observation_tick=0,calibration_inference_ms=2.)

    def infer(self,request):
        return dict(actions=[[0.]*self.action_dim]*8,observation_tick=request["observation_tick"],inference_ms=2.)


class TaskTests(unittest.TestCase):
    def test_level_hash_matches_across_windows_and_linux(self):
        original = b'{\n  "env_state": {"position": [1, 2]}\n}\n'
        with tempfile.TemporaryDirectory() as temp:
            linux,windows = Path(temp)/"linux.json",Path(temp)/"windows.json"
            linux.write_bytes(original)
            windows.write_bytes(original.replace(b"\n",b"\r\n"))
            self.assertEqual(level_sha256(linux),level_sha256(windows))
            self.assertEqual(level_sha256(windows),hashlib.sha256(original).hexdigest())
            windows.write_bytes(original.replace(b'"env_state"',b'"changed_state"',1))
            self.assertNotEqual(level_sha256(linux),level_sha256(windows))

    def test_task_config_validation_and_legacy_level(self):
        self.assertEqual(selected_tasks({"level":TASKS["grasp_easy"]})[0]["name"],"grasp_easy")
        self.assertEqual(len(selected_tasks({"tasks":list(TASKS)})),5)
        for tasks in [[],["unknown"],["grasp_easy","grasp_easy"],"grasp_easy",[{}]]:
            with self.subTest(tasks=tasks),self.assertRaises(ValueError):
                selected_tasks({"tasks":tasks})

    def test_registry_routes_and_caches_matching_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            for task in ["grasp_easy","cartpole_thrust"]:
                task_checkpoint(temp,task).write_bytes(b"test")
            factory = unittest.mock.Mock(side_effect=Policy)
            registry = PolicyRegistry(temp,factory)
            self.assertEqual(factory.call_count,0)
            with self.assertRaises(ValueError):
                registry.infer({"task":"grasp_easy"})
            for task in ["grasp_easy","cartpole_thrust","grasp_easy"]:
                description = registry.select_task(task)
                self.assertEqual(description["checkpoint_sha256"],f"hash-{task}")
                self.assertEqual(len(description["available_tasks"]),2)
            self.assertEqual(factory.call_count,2)
            with self.assertRaises(ValueError):
                registry.infer({"task":"cartpole_thrust"})
            with self.assertRaises(ValueError):
                registry.select_task("../escape")

    def test_average_is_macro_average_and_incomplete_sweeps_fail(self):
        rows = [dict(task="grasp_easy",method="rtc",predictor="p95",predictor_basis="inference_ms",
                     episodes=10,solve_rate=0.,buffer_exhaustion_steps=0,deadline_overrun_steps=0),
                dict(task="cartpole_thrust",method="rtc",predictor="p95",predictor_basis="inference_ms",
                     episodes=100,solve_rate=1.,buffer_exhaustion_steps=0,deadline_overrun_steps=0)]
        summary = aggregate_tasks(rows,["grasp_easy","cartpole_thrust"],[("rtc","p95")])
        self.assertEqual(summary["overall"][0]["average_solve_rate"],.5)
        with self.assertRaises(ValueError):
            aggregate_tasks(rows[:1],["grasp_easy","cartpole_thrust"],[("rtc","p95")])

    def test_complete_two_task_three_mode_sweep_over_tcp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoints = root/"policies"
            checkpoints.mkdir()
            for task in ["grasp_easy","cartpole_thrust"]:
                task_checkpoint(checkpoints,task).write_bytes(b"test")
            server = make_server("127.0.0.1",0,PolicyRegistry(checkpoints,Policy))
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            config = json.loads(Path("configs/rtc_remote.json").read_text())
            config.update(tasks=["grasp_easy","cartpole_thrust"],episodes=2,control_ms=5,
                          predictors={"last":{"kind":"last"}})
            try:
                with patch("wireless_rtc.simulation.kinetix.KinetixSimulator",side_effect=Simulator),patch("sys.stdout"):
                    run_remote(config,"127.0.0.1",server.server_address[1],root/"run",{})
                summary = json.loads((root/"run"/"summary.json").read_text())
                self.assertEqual(len(summary["per_task"]),6)
                self.assertEqual(len(summary["overall"]),3)
                self.assertTrue(all(row["average_solve_rate"] == .5 for row in summary["overall"]))
                self.assertTrue((root/"run"/"summary.csv").is_file())
                manifest = json.loads((root/"run"/"manifest.json").read_text())
                self.assertEqual(manifest["status"],"completed")
                for task in config["tasks"]:
                    self.assertEqual(manifest["tasks"][task]["server"]["checkpoint_sha256"],f"hash-{task}")
                    for method in CONTROL_MODES:
                        name = "naive" if method == "naive" else f"{method}-last"
                        with (root/"run"/"tasks"/task/name/"requests.csv").open() as f:
                            rows = list(csv.DictReader(f))
                        self.assertTrue(rows)
                        self.assertTrue(all(row["task"] == task for row in rows))
                        successful = [row for row in rows if row["status"] == "ok"]
                        measured = [float(row["actual_ms"]) for row in successful]
                        condition = next(row for row in summary["per_task"]
                                         if row["task"] == task and row["method"] == method)
                        self.assertAlmostEqual(condition["average_rtt_ms"],sum(measured)/len(measured))
                        self.assertEqual(condition["rtt_samples"],len(measured))
                        self.assertEqual(condition["average_injected_delay_ms"],0.)
                        if method == "naive":
                            self.assertIsNone(condition["prediction_mae_ms"])
                            self.assertEqual(condition["prediction_samples"],0)
                        else:
                            target = "inference_ms" if method == "rtc" else "actual_ms"
                            expected_mae = sum(abs(float(row["predicted_ms"])-float(row[target]))
                                               for row in successful)/len(successful)
                            self.assertAlmostEqual(condition["prediction_mae_ms"],expected_mae)
                        with (root/"run"/"tasks"/task/name/"episodes.csv").open() as f:
                            episode_rows = list(csv.DictReader(f))
                        for episode_row in episode_rows:
                            episode_samples = [float(row["actual_ms"]) for row in successful
                                               if row["episode"] == episode_row["episode"]]
                            self.assertAlmostEqual(float(episode_row["average_rtt_ms"]),
                                                   sum(episode_samples)/len(episode_samples))
                with (root/"run"/"summary.csv").open() as f:
                    csv_summary = list(csv.DictReader(f))
                for row, overall in zip(csv_summary, summary["overall"]):
                    self.assertEqual(float(row["average_rtt_ms"]),overall["average_rtt_ms"])
                    self.assertEqual(float(row["average_injected_delay_ms"]),0.)
                    if row["method"] == "naive":
                        self.assertEqual(row["prediction_mae_ms"],"")
                    else:
                        self.assertEqual(float(row["prediction_mae_ms"]),overall["prediction_mae_ms"])
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

    def test_missing_checkpoint_fails_before_any_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            task_checkpoint(temp,"grasp_easy").write_bytes(b"test")
            registry = PolicyRegistry(temp,Policy)
            server = make_server("127.0.0.1",0,registry)
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            config = json.loads(Path("configs/rtc_remote.json").read_text())
            try:
                with patch("wireless_rtc.simulation.kinetix.KinetixSimulator") as simulator:
                    with self.assertRaisesRegex(ValueError,"missing tasks"):
                        run_remote(config,"127.0.0.1",server.server_address[1],Path(temp)/"run",{})
                    simulator.assert_not_called()
                self.assertFalse((Path(temp)/"run"/"summary.json").exists())
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)

    def test_compatibility_error_records_differing_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            task_checkpoint(temp,"grasp_easy").write_bytes(b"test")
            server = make_server("127.0.0.1",0,PolicyRegistry(temp,Policy))
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            config = json.loads(Path("configs/rtc_remote.json").read_text())
            config["tasks"] = ["grasp_easy"]
            simulator = Simulator(TASKS["grasp_easy"],42)
            simulator.level_sha256 = "changed-level"
            try:
                with patch("wireless_rtc.simulation.kinetix.KinetixSimulator",return_value=simulator):
                    with self.assertRaisesRegex(ValueError,"level_sha256.*changed-level"):
                        run_remote(config,"127.0.0.1",server.server_address[1],Path(temp)/"run",{})
                manifest = json.loads((Path(temp)/"run"/"manifest.json").read_text())
                mismatches = manifest["tasks"]["grasp_easy"]["compatibility_mismatches"]
                self.assertEqual(mismatches["level_sha256"]["client"],"changed-level")
                self.assertEqual(manifest["tasks"]["grasp_easy"]["status"],"error")
                self.assertFalse((Path(temp)/"run"/"summary.json").exists())
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)
