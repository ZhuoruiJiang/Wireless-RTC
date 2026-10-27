"""Exercise real HTTP streams and ensure rendering never queues controller states."""
import http.client
import json
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from wireless_rtc.cli import main
from wireless_rtc.simulation.viewer import SimulationViewer


class ViewerTests(unittest.TestCase):
    def test_page_status_and_multipart_frame(self):
        jpeg = b"\xff\xd8test-frame\xff\xd9"
        with SimulationViewer(port=0, fps=60) as viewer:
            base = f"http://127.0.0.1:{viewer.port}"
            with urlopen(base, timeout=3) as response:
                self.assertIn(b'src="/stream.mjpg"', response.read())
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            viewer.set_renderer(lambda state: jpeg)
            viewer.update(object(), phase="Running", tick=12, success=False)
            with urlopen(base+"/status.json", timeout=3) as response:
                status = json.load(response)
                self.assertEqual(status["tick"], 12)
                self.assertEqual(status["phase"], "Running")
            connection = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=3)
            try:
                connection.request("GET", "/stream.mjpg")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn("boundary=frame", response.getheader("Content-Type"))
                self.assertEqual(response.readline(), b"--frame\r\n")
                self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
                length = int(response.readline().split(b":")[1])
                self.assertEqual(response.readline(), b"\r\n")
                self.assertEqual(response.read(length), jpeg)
            finally:
                connection.close()

    def test_slow_renderer_drops_intermediate_states(self):
        started, release, latest_rendered = threading.Event(), threading.Event(), threading.Event()
        seen = []

        def renderer(state):
            seen.append(state)
            if state == 0:
                started.set()
                if not release.wait(3):
                    raise TimeoutError("test did not release renderer")
            if state == 1000:
                latest_rendered.set()
            return b"jpeg"

        with SimulationViewer(port=0, fps=60) as viewer:
            viewer.set_renderer(renderer)
            viewer.update(0)
            self.assertTrue(started.wait(3))
            try:
                # Producer keeps running even with a renderer deliberately blocked.
                for state in range(1,1001):
                    viewer.update(state, tick=state)
                self.assertEqual(viewer.status()["tick"], 1000)
            finally:
                release.set()
            self.assertTrue(latest_rendered.wait(3))
            self.assertEqual(seen, [0,1000])

    def test_render_failure_is_visible_without_crashing_producer(self):
        with SimulationViewer(port=0) as viewer:
            def broken(state):
                raise RuntimeError("missing renderer dependency")
            viewer.set_renderer(broken)
            viewer.update(object())
            viewer._render_thread.join(timeout=3)
            self.assertFalse(viewer._render_thread.is_alive())
            self.assertIn("missing renderer dependency", viewer.status()["viewer_error"])
            viewer.update(tick=50)
            self.assertEqual(viewer.status()["tick"], 50)

    def test_close_wakes_browser_waiting_for_first_frame(self):
        viewer = SimulationViewer(port=0)
        connection = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=3)
        try:
            connection.request("GET", "/stream.mjpg")
            response = connection.getresponse()
            viewer.close()
            self.assertEqual(response.read(), b"")
            self.assertFalse(viewer._render_thread.is_alive())
            self.assertFalse(viewer._http_thread.is_alive())
        finally:
            connection.close()

    def test_invalid_cli_viewer_options_fail_before_loading_policy(self):
        for options in [["--view-pace"], ["--view-fps", "nan"], ["--view-port", "0"]]:
            with self.subTest(options=options), patch("sys.stderr"), self.assertRaises(SystemExit) as result:
                main(["rtc", "--checkpoint", "missing.pkl", "--output", "unused", *options])
            self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
