"""Actual pretrained upstream RTC smoke test, enabled explicitly with RTC_CHECKPOINT."""
import os
import io
import csv
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("RTC_CHECKPOINT"),"Set RTC_CHECKPOINT after installing RTC dependencies")
class PretrainedTests(unittest.TestCase):
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
                connection = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=30)
                try:
                    connection.request("GET", "/stream.mjpg")
                    response = connection.getresponse()
                    self.assertEqual(response.readline(), b"--frame\r\n")
                    self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
                    size = int(response.readline().split(b":")[1])
                    response.readline()
                    from PIL import Image
                    with Image.open(io.BytesIO(response.read(size))) as image:
                        image.verify()
                finally:
                    connection.close()
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
