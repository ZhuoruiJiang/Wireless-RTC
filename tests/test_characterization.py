import csv
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wireless_rtc.network.characterization import export_table, prepare_workload, run_characterization
from wireless_rtc.server.service import make_server


class FakePolicy:
    def describe(self):
        return dict(task="cartpole_thrust", level_sha256="level", observation_dim=2,
                    action_dim=1, horizon=8, checkpoint_sha256="checkpoint")

    def select_task(self, task):
        return self.describe()

    def initialize(self, request):
        return dict(task="cartpole_thrust", observation_tick=0, actions=[[0.0]] * 8)

    def infer(self, request):
        if request["observation"] != [0.2, 0.4]:
            raise ValueError("Invalid observation")
        return dict(task="cartpole_thrust", observation_tick=0, actions=[[0.1]] * 8, inference_ms=.1)


class FakeSimulator:
    def __init__(self, *args):
        self.level_sha256, self.observation_dim, self.action_dim = "level", 2, 1
        self.jax = SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed))
        self.np = SimpleNamespace(asarray=lambda obs: SimpleNamespace(tolist=lambda: obs))

    def reset(self, key):
        return [0.2, 0.4], None


class CharacterizationTests(unittest.TestCase):
    def test_response_duration_uses_high_resolution_clock(self):
        client = SimpleNamespace(socket=SimpleNamespace(settimeout=lambda t: None), counter=0,
                                 close=lambda: None, request=lambda *a, **kw: ({"server_ms": 1}, 1.5))
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"), (
                patch("wireless_rtc.network.characterization.RPCClient", return_value=client)), (
                patch("wireless_rtc.network.characterization.prepare_workload",
                      return_value=("probe", {}, lambda r: None, dict(workload="echo")))), (
                patch("wireless_rtc.network.characterization.time.monotonic_ns", side_effect=range(0, 200_000_000, 10_000_000))), (
                patch("wireless_rtc.network.characterization.time.perf_counter_ns", side_effect=[100_000_000, 101_500_000])):
            path = run_characterization("localhost", 1234, Path(temp)/"run", setting="stationary",
                                        split="development", wifi=False, duration_s=.025, workload="echo")
            with (path/"trace.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["actual_ms"]), 1.5)

    def test_policy_workload_through_actual_rpc_and_separate_failures(self):
        server = make_server("127.0.0.1", 0, FakePolicy())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp, patch("wireless_rtc.simulation.kinetix.KinetixSimulator", FakeSimulator), patch("sys.stdout"):
                path = run_characterization("127.0.0.1", server.server_address[1], Path(temp)/"run",
                    setting="walking", split="test", duration_s=.08, interval_ms=2, warmup=2, wifi=False)
                manifest = json.loads((path/"manifest.json").read_text())
                summary = json.loads((path/"summary.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["workload"]["workload"], "policy")
                self.assertGreater(summary["completed_requests"], 0)
                self.assertIsNone(summary["median_rssi_dbm"])
                with (path/"trace.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertTrue(all(float(r["request_bytes"]) > 0 and float(r["response_bytes"]) > 0 for r in rows))
                self.assertTrue(all(int(r["request_start_monotonic_ns"]) >= manifest["measurement_start_ns"] for r in rows))
                with self.assertRaises(FileExistsError):
                    run_characterization("127.0.0.1", server.server_address[1], path,
                        setting="walking", split="test", duration_s=.01, wifi=False)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_reject_injected_policy_delay_before_recording(self):
        client = SimpleNamespace(request=lambda *a, **kw: (dict(response_delay=dict(kind="pattern")), 1))
        with patch("wireless_rtc.simulation.kinetix.KinetixSimulator", FakeSimulator):
            with self.assertRaisesRegex(ValueError, "--delay none"):
                prepare_workload(client, "policy", "cartpole_thrust", 1024, 42, 2)

    def test_interrupt_retains_partial_trace_and_status(self):
        client = SimpleNamespace(socket=SimpleNamespace(settimeout=lambda t: None),
                                 counter=0, close=lambda: None,
                                 request=lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"), (
                patch("wireless_rtc.network.characterization.RPCClient", return_value=client)), (
                patch("wireless_rtc.network.characterization.prepare_workload",
                      return_value=("probe", {}, lambda r: None, dict(workload="echo")))):
            path = run_characterization("localhost", 1234, Path(temp)/"run", setting="stationary",
                                        split="test", wifi=False, duration_s=.1, workload="echo")
            summary = json.loads((path/"summary.json").read_text())
            self.assertEqual(summary["status"], "interrupted")

    def make_run(self, root, setting, name, delays, rssi, status="complete", split="test", workload="policy"):
        path = root/setting/split/name
        path.mkdir(parents=True)
        manifest = dict(kind="network_characterization", setting=setting, split=split, status=status,
                        workload=dict(workload=workload, task="cartpole_thrust"), host="localhost", port=8765,
                        interval_ms=80, measurement_start_ns=100, measurement_end_ns=200)
        (path/"manifest.json").write_text(json.dumps(manifest))
        with (path/"trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["status", "actual_ms"])
            writer.writeheader()
            writer.writerows([dict(status="ok", actual_ms=d) for d in delays]+[dict(status="error", actual_ms="")])
        with (path/"wifi.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["wifi_available_monotonic_ns", "wifi_status", "wifi_rssi_dbm"])
            writer.writeheader()
            writer.writerow(dict(wifi_available_monotonic_ns=50, wifi_status="ok", wifi_rssi_dbm=-100))
            for value in rssi:
                writer.writerow(dict(wifi_available_monotonic_ns=150, wifi_status="ok", wifi_rssi_dbm=value))
        return path

    def test_report_pools_raw_delays_and_independent_polls_excludes_partial_and_development(self):
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"):
            root = Path(temp)/"runs"
            self.make_run(root, "stationary", "a", [10]*20, [-40])
            self.make_run(root, "stationary", "b", [100], [-80, -80, -80])
            self.make_run(root, "walking", "c", [20, 30], [])
            self.make_run(root, "walking", "partial", [999], [-5], status="interrupted")
            self.make_run(root, "walking", "dev", [999], [-5], split="development")
            self.make_run(root, "walking", "echo", [999], [-5], workload="echo")
            output = export_table(root, Path(temp)/"report")
            report = json.loads((output/"summary.json").read_text())
            self.assertEqual(len(report["included_runs"]), 3)
            self.assertEqual(len(report["excluded_runs"]), 1)
            first, second = report["table"]
            self.assertEqual(first["completed_requests"], 21)
            self.assertEqual(first["response_p50_ms"], 10)
            self.assertEqual(first["median_rssi_dbm"], -80)
            self.assertIsNone(second["median_rssi_dbm"])
            self.assertEqual(report["audits"]["stationary"]["failed_requests"], 2)
            self.assertIn("29.50", (output/"table1.csv").read_text())
            self.assertIn(r"\textemdash", (output/"table1.tex").read_text())
            with self.assertRaises(FileExistsError):
                export_table(root, output)

    def test_report_rejects_different_workload_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"runs"
            self.make_run(root, "stationary", "a", [10], [-40])
            other = self.make_run(root, "walking", "b", [20], [-60])
            path = other/"manifest.json"
            manifest = json.loads(path.read_text())
            manifest["interval_ms"] = 100
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "differ"):
                export_table(root, Path(temp)/"report")

    def test_strong_weak_export_excludes_exploratory_walking(self):
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"):
            root = Path(temp)/"runs"
            self.make_run(root, "strong", "a", [10, 12], [-50])
            self.make_run(root, "weak", "b", [30, 60], [-83])
            self.make_run(root, "walking", "exploratory", [999], [-95])
            output = export_table(root, Path(temp)/"report", settings=["strong", "weak"])
            report = json.loads((output/"summary.json").read_text())
            self.assertEqual([row["setting"] for row in report["table"]], ["strong", "weak"])
            self.assertEqual(len(report["included_runs"]), 2)
            self.assertEqual(report["missing_settings"], [])
            self.assertIn("Weak Wi-Fi", (output/"table1.tex").read_text())

    def test_invalid_duration_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"run"
            for value in [0, -1, float("nan")]:
                with self.assertRaises(ValueError):
                    run_characterization("localhost", 1234, path, setting="walking", split="test", duration_s=value, wifi=False)
            self.assertFalse(path.exists())

    def test_explicit_config_ignores_unlisted_runs_and_stale_legacy_map(self):
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"):
            root = Path(temp)/"runs"
            strong = self.make_run(root,"stationary","selected-strong",[10,20],[-50])
            weak = self.make_run(root,"weak","selected-weak",[30,60],[-81],split="development")
            self.make_run(root,"weak","unlisted",[999],[-99])
            (root/"session_map.json").write_text(json.dumps(dict(schema_version=1,settings={"deleted/test/run":"weak"})))
            config = root/"selection.json"
            config.write_text(json.dumps(dict(schema_version=1,runs={
                "strong":[strong.relative_to(root).as_posix()],"weak":[weak.relative_to(root).as_posix()]})))
            before = (weak/"manifest.json").read_bytes()
            output = export_table(root,Path(temp)/"report",selection_config=config)
            report = json.loads((output/"summary.json").read_text())
            self.assertEqual(len(report["included_runs"]),2)
            self.assertEqual(report["table"][1]["response_p50_ms"],45)
            self.assertIsNone(report["session_map"])
            self.assertEqual(report["session_assignments"][1]["saved_split"],"development")
            self.assertEqual((weak/"manifest.json").read_bytes(),before)
            self.assertTrue((output/"selection.json").exists())

    def test_selected_incomplete_run_fails_instead_of_being_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"runs"
            path = self.make_run(root,"weak","unfinished",[10],[-81],status="interrupted")
            config = root/"selection.json"
            config.write_text(json.dumps(dict(schema_version=1,runs={"weak":[path.relative_to(root).as_posix()]})))
            with self.assertRaisesRegex(ValueError,"not complete"):
                export_table(root,Path(temp)/"report",selection_config=config)
            self.assertFalse((Path(temp)/"report").exists())
            manifest = path/"manifest.json"
            document = json.loads(manifest.read_text())
            document["status"] = "complete"
            manifest.write_text(json.dumps(document))
            (path/"trace.csv").unlink()
            with self.assertRaisesRegex(ValueError,"missing trace.csv"):
                export_table(root,Path(temp)/"report",selection_config=config)

    def test_session_mapping_uses_operator_labels_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as temp, patch("sys.stdout"):
            root = Path(temp)/"runs"
            a = self.make_run(root, "stationary", "desk", [10, 12], [-60])
            b = self.make_run(root, "walking", "outdoor", [30, 60], [-83])
            original = (b/"manifest.json").read_bytes()
            (root/"session_map.json").write_text(json.dumps(dict(schema_version=1, settings={
                a.relative_to(root).as_posix(): "strong", b.relative_to(root).as_posix(): "weak"})))
            output = export_table(root, Path(temp)/"report", settings=["strong", "weak"])
            report = json.loads((output/"summary.json").read_text())
            self.assertEqual([r["setting"] for r in report["table"]], ["strong", "weak"])
            self.assertEqual(report["session_assignments"][1]["saved_setting"], "walking")
            self.assertEqual((b/"manifest.json").read_bytes(), original)
            (root/"session_map.json").write_text(json.dumps(dict(schema_version=1, settings={"missing/run": "weak"})))
            with self.assertRaisesRegex(ValueError, "missing session"):
                export_table(root, Path(temp)/"other", settings=["strong", "weak"])
