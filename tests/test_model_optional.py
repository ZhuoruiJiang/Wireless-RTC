"""Actual pretrained upstream RTC smoke test, enabled explicitly with RTC_CHECKPOINT."""
import os
import io
import csv
import base64
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("RTC_CHECKPOINT"),"Set RTC_CHECKPOINT after installing RTC dependencies")
class PretrainedTests(unittest.TestCase):
    def test_live_rpc_with_independent_client_simulator(self):
        from wireless_rtc.client.live import run_remote
        from wireless_rtc.server.service import make_server
        from wireless_rtc.server.delay import NormalResponseDelay
        from wireless_rtc.simulation.kinetix import KinetixPolicy, KinetixSimulator
        from wireless_rtc.simulation.viewer import SimulationViewer
        policy = KinetixPolicy(os.environ["RTC_CHECKPOINT"])
        simulator = KinetixSimulator()
        self.assertFalse(hasattr(simulator,"policy"))
        simulator.params = simulator.params.replace(max_timesteps=8)
        config = json.loads(Path("configs/rtc_remote.json").read_text())
        config.update(episodes=1,methods=["rtc","naive"],
                      predictors={"recent_max": config["predictors"]["recent_max"]})
        server = make_server("127.0.0.1",0,policy,
                             response_delay=NormalResponseDelay(mean_ms=25,std_ms=5,max_ms=100))
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp, SimulationViewer(port=0) as viewer, patch(
                    "wireless_rtc.simulation.kinetix.KinetixSimulator",return_value=simulator):
                root = Path(temp)/"live"
                run_remote(config,"127.0.0.1",server.server_address[1],root,{},viewer=viewer)
                manifest = json.loads((root/"manifest.json").read_text())
                self.assertEqual(manifest["status"],"completed")
                self.assertEqual(manifest["server"]["checkpoint_sha256"],policy.checkpoint_sha256)
                self.assertEqual(manifest["server"]["response_delay"]["kind"],"clipped_normal")
                self.assertTrue(viewer.status()["phase"].startswith("Completed"))
                for method in config["methods"]:
                    directory = root/f"{method}-recent_max"
                    with (directory/"requests.csv").open() as f:
                        rows = list(csv.DictReader(f))
                    self.assertTrue(rows)
                    self.assertTrue(all(r["status"] == "ok" and float(r["actual_ms"]) > 0 for r in rows))
                    self.assertTrue(all(float(r["injected_delay_ms"]) > 0 for r in rows))
                    with (directory/"ticks.csv").open() as f:
                        self.assertEqual(len(list(csv.DictReader(f))),8)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)

    def test_actual_checkpoint_generates_finite_chunks(self):
        from wireless_rtc.simulation.kinetix import KinetixPolicy
        p=KinetixPolicy(os.environ["RTC_CHECKPOINT"])
        jax,jnp,np=p.jax,p.jnp,p.np
        obs,state=p.reset(jax.random.key(0))
        next_obs,_,_,_,_=p.step(jax.random.key(3),state,jnp.zeros(p.action_dim))
        self.assertTrue(np.isfinite(np.asarray(next_obs)).all())
        previous=p.plain(jax.random.key(1),obs[None])
        for d in [0,2,4]:
            new=p.guided(jax.random.key(2),obs[None],previous,d,4)
            self.assertEqual(new.shape,previous.shape)
            self.assertTrue(np.isfinite(np.asarray(new)).all())

    def test_headless_frame_is_a_valid_jpeg(self):
        from PIL import Image
        from wireless_rtc.simulation.kinetix import KinetixPolicy
        p = KinetixPolicy(os.environ["RTC_CHECKPOINT"])
        _, state = p.reset(p.jax.random.key(0))
        jpeg = p.make_frame_renderer()(state)
        with Image.open(io.BytesIO(jpeg)) as frame:
            self.assertEqual(frame.format, "JPEG")
            self.assertEqual(frame.size, (256,256))
            self.assertEqual(frame.mode, "RGB")

    def test_viewing_preserves_short_rtc_run(self):
        from wireless_rtc.simulation.kinetix import KinetixPolicy, run_experiment
        from wireless_rtc.simulation.viewer import SimulationViewer
        checkpoint = os.environ["RTC_CHECKPOINT"]
        backend = KinetixPolicy(checkpoint)
        # Eight real physics steps and two policy requests, with the actual scheduler.
        backend.params = backend.params.replace(max_timesteps=8)
        config = json.loads(Path("configs/rtc_fixed.json").read_text())
        config.update(episodes=1, methods=["rtc"])
        with tempfile.TemporaryDirectory() as temp, patch(
                "wireless_rtc.simulation.kinetix.KinetixPolicy", return_value=backend):
            root = Path(temp)
            run_experiment(config, checkpoint, root/"plain", {})
            with SimulationViewer(port=0) as viewer:
                run_experiment(config, checkpoint, root/"view", {}, viewer=viewer)
                self.assertTrue(viewer.status()["phase"].startswith("Completed"))
                deadline = time.monotonic()+30
                while not viewer.status()["frame_ready"] and time.monotonic() < deadline:
                    time.sleep(.02)
                snapshot = viewer.snapshot()
                self.assertIsNotNone(snapshot["frame"])
                from PIL import Image
                with Image.open(io.BytesIO(base64.b64decode(snapshot["frame"]["jpeg"]))) as image:
                    image.verify()
            for filename in ["episodes.csv", "requests.csv"]:
                conditions = []
                for directory in ["plain", "view"]:
                    with (root/directory/"rtc-recent_max"/filename).open() as f:
                        rows = list(csv.DictReader(f))
                    for row in rows:
                        row.pop("host_inference_ms", None)
                    conditions.append(rows)
                self.assertEqual(*conditions)
            manifest = json.loads((root/"view"/"manifest.json").read_text())
            self.assertTrue(manifest["viewer"]["enabled"])


if __name__=="__main__":
    unittest.main()
