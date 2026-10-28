"""Public HTTP and concurrency contracts for the canvas viewer."""
import base64
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

from wireless_rtc.cli import main
from wireless_rtc.simulation.viewer import SimulationViewer

JPEG = b"\xff\xd8test-image\xff\xd9"


def eventually(predicate, timeout=3):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    raise AssertionError("Timed out waiting for viewer state")


class ViewerTests(unittest.TestCase):
    def test_port_cannot_silently_route_to_an_older_viewer(self):
        with SimulationViewer(port=0) as first:
            with self.assertRaises(OSError):
                SimulationViewer(port=first.port)
            with urlopen(f"http://127.0.0.1:{first.port}/api/snapshot",timeout=3) as response:
                self.assertEqual(json.load(response)["session"],first.snapshot()["session"])

    def test_packaged_page_and_assets_are_served(self):
        with SimulationViewer(port=0) as viewer:
            for path,content_type,marker in [("/","text/html",b'<canvas id="scene"'),
                                             ("/viewer.js","text/javascript",b"createImageBitmap"),
                                             ("/viewer.css","text/css",b".workspace")]:
                with urlopen(f"http://127.0.0.1:{viewer.port}{path}",timeout=3) as response:
                    self.assertIn(content_type,response.headers["Content-Type"])
                    self.assertEqual(response.headers["Cache-Control"],"no-store")
                    self.assertIn(marker,response.read())
            with self.assertRaises(HTTPError) as error:
                urlopen(f"http://127.0.0.1:{viewer.port}/../pyproject.toml",timeout=3)
            self.assertEqual(error.exception.code,404)
            error.exception.close()

    def test_api_startup_final_frame_cursors_and_restart(self):
        with SimulationViewer(port=0) as viewer:
            base = f"http://127.0.0.1:{viewer.port}/api/snapshot"
            with urlopen(base,timeout=3) as response:
                initial = json.load(response)
            self.assertIsNone(initial["frame"])
            self.assertEqual(initial["renderer"]["phase"],"waiting")
            viewer.set_renderer(lambda state:JPEG)
            viewer.update(0,tick=42,phase="Completed")
            eventually(lambda:viewer.status()["frame_ready"])
            with urlopen(base,timeout=3) as response:
                snapshot = json.load(response)
            frame,session = snapshot["frame"],snapshot["session"]
            self.assertEqual(base64.b64decode(frame["jpeg"]),JPEG)
            self.assertEqual(frame["state"]["tick"],42)
            exposed = viewer.snapshot()
            exposed["frame"]["state"]["tick"] = -1
            self.assertEqual(viewer.snapshot()["frame"]["state"]["tick"],42)
            with urlopen(f"{base}?session={session}&after={frame['sequence']}",timeout=3) as response:
                self.assertIsNone(json.load(response)["frame"])
            # An old browser must receive a final image even if its old cursor is higher.
            with urlopen(f"{base}?session=previous-server&after=999999",timeout=3) as response:
                self.assertEqual(json.load(response)["frame"],frame)
            for value in ["-1","nan","abc"]:
                with self.assertRaises(HTTPError) as error:
                    urlopen(f"{base}?after={value}",timeout=3)
                self.assertEqual(error.exception.code,400)
                error.exception.close()
        with SimulationViewer(port=0) as restarted:
            self.assertNotEqual(restarted.snapshot()["session"],session)

    def test_slow_rendering_keeps_latest_state_and_matching_metadata(self):
        started,release = threading.Event(),threading.Event()
        seen = []
        def render(state):
            seen.append(state)
            if state == 0:
                started.set()
                if not release.wait(3):
                    raise TimeoutError("Test did not release renderer")
            return JPEG
        with SimulationViewer(port=0,fps=60) as viewer:
            viewer.set_renderer(render)
            viewer.update(0,tick=0)
            self.assertTrue(started.wait(3))
            try:
                for tick in range(1,1001):
                    viewer.update(tick,tick=tick)
                self.assertEqual(viewer.snapshot()["state"]["tick"],1000)
            finally:
                release.set()
            snapshot = eventually(lambda: (s if (s:=viewer.snapshot())["frame"] and s["frame"]["state"]["tick"] == 1000 else None))
            self.assertEqual(seen,[0,1000])
            self.assertEqual(snapshot["frame"]["state"]["tick"],1000)
            viewer.update(tick=1001)
            self.assertEqual(viewer.snapshot()["state"]["tick"],1001)
            self.assertEqual(viewer.snapshot()["frame"]["state"]["tick"],1000)

    def test_render_errors_are_visible_and_renderer_can_recover(self):
        with SimulationViewer(port=0) as viewer:
            def broken(state):
                raise RuntimeError("Renderer unavailable")
            viewer.set_renderer(broken)
            viewer.update(0,tick=1)
            eventually(lambda:viewer.snapshot()["renderer"]["error"])
            self.assertIn("Renderer unavailable",viewer.snapshot()["renderer"]["error"])
            viewer.update(tick=2)
            self.assertEqual(viewer.status()["tick"],2)
            viewer.set_renderer(lambda state:JPEG)
            eventually(lambda:viewer.status()["frame_ready"])
            self.assertIsNone(viewer.snapshot()["renderer"]["error"])

    def test_incomplete_image_is_reported_and_close_is_idempotent(self):
        viewer = SimulationViewer(port=0)
        viewer.set_renderer(lambda state:b"not-an-image")
        viewer.update(0)
        eventually(lambda:viewer.snapshot()["renderer"]["error"])
        self.assertIn("complete JPEG",viewer.status()["viewer_error"])
        viewer.close()
        viewer.close()
        self.assertFalse(viewer._http_thread.is_alive())
        self.assertFalse(viewer._render_thread.is_alive())

    def test_invalid_and_occupied_ports_fail_before_policy_load(self):
        for options in [["--view-pace"],["--view-fps","nan"],["--view-port","0"]]:
            with self.subTest(options=options),patch("sys.stderr"),self.assertRaises(SystemExit) as result:
                main(["rtc","--checkpoint","missing.pkl","--output","unused",*options])
            self.assertEqual(result.exception.code,2)
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1",0))
            occupied.listen()
            with patch("sys.stderr"),self.assertRaises(SystemExit) as result:
                main(["rtc","--checkpoint","missing.pkl","--output","unused","--view",
                      "--view-port",str(occupied.getsockname()[1])])
            self.assertEqual(result.exception.code,2)


if __name__ == "__main__":
    unittest.main()
