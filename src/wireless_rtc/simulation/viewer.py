"""Local canvas viewer with isolated rendering and atomic HTTP snapshots."""
import base64
from dataclasses import dataclass
from importlib.resources import files
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import uuid


@dataclass(frozen=True)
class Frame:
    sequence: int
    jpeg: bytes
    state: dict
    rendered_at: float
    render_ms: float


class ViewerHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can route requests to an old viewer even after a
    # second viewer successfully binds. Require exclusive ownership there.
    allow_reuse_address = not hasattr(socket,"SO_EXCLUSIVEADDRUSE")

    def server_bind(self):
        if hasattr(socket,"SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


class SimulationViewer:
    """One pending immutable state, one completed image, no frame backlog.

    Physics only publishes references. Rendering, encoding, and browser traffic
    run separately. Every image retains metadata from the state it depicts.
    """

    def __init__(self, port=8080, fps=10):
        if not math.isfinite(fps) or not 0 < fps <= 60:
            raise ValueError("Viewer fps must be in (0,60]")
        self.fps = fps
        self._session = uuid.uuid4().hex
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._renderer = None
        self._generation = 0
        self._state = None
        self._version = 0
        self._frame = None
        self._sequence = 0
        self._render_phase = "waiting"
        self._render_error = None
        self._status = {"phase": "Starting simulation"}
        assets = files("wireless_rtc.simulation").joinpath("web")
        routes = {"/": (assets.joinpath("index.html").read_bytes(), "text/html; charset=utf-8"),
                  "/viewer.css": (assets.joinpath("viewer.css").read_bytes(), "text/css; charset=utf-8"),
                  "/viewer.js": (assets.joinpath("viewer.js").read_bytes(), "text/javascript; charset=utf-8")}
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.connection.settimeout(5)
                parsed = urlsplit(self.path)
                code = 200
                if parsed.path in routes:
                    body, content_type = routes[parsed.path]
                elif parsed.path == "/api/snapshot":
                    query = parse_qs(parsed.query)
                    try:
                        after = int(query.get("after", ["0"])[0])
                        if after < 0:
                            raise ValueError
                        payload = viewer.snapshot(after, query.get("session", [None])[0])
                    except ValueError:
                        code, payload = 400, {"error": "after must be a nonnegative integer"}
                    body, content_type = json.dumps(payload, allow_nan=False).encode(), "application/json"
                else:
                    code, body, content_type = 404, b"Not found", "text/plain; charset=utf-8"
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; frame-ancestors 'none'")
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass

        self._server = ViewerHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self._http_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._http_thread.start()
        self._render_thread.start()

    @property
    def port(self):
        return self._server.server_address[1]

    def set_renderer(self, renderer, reset=False):
        with self._condition:
            if reset:
                self._state = None
                self._frame = None
            self._renderer = renderer
            self._generation += 1
            self._version += 1
            self._render_error = None
            self._render_phase = "waiting"
            self._condition.notify_all()

    def update(self, state=None, **status):
        with self._condition:
            self._status.update(status)
            if state is not None:
                self._state = state
                self._version += 1
            self._condition.notify_all()

    def status(self):
        with self._condition:
            return dict(self._status, viewer_error=self._render_error, frame_ready=self._frame is not None)

    def snapshot(self, after=0, session=None):
        """Copy coherent state under the lock; encode images outside that lock."""
        with self._condition:
            frame = self._frame
            payload = dict(session=self._session, fps=self.fps, state=dict(self._status),
                           renderer=dict(phase=self._render_phase, error=self._render_error), frame=None)
        if frame is not None:
            payload["renderer"].update(sequence=frame.sequence, render_ms=frame.render_ms,
                                      age_ms=max(0,(time.monotonic()-frame.rendered_at)*1000))
            if session != self._session or frame.sequence > after:
                payload["frame"] = dict(sequence=frame.sequence, state=dict(frame.state),
                                        jpeg=base64.b64encode(frame.jpeg).decode("ascii"))
        return payload

    def _render_loop(self):
        consumed, next_frame = -1, 0.
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._stop.is_set() or (
                    self._renderer is not None and self._state is not None and self._version > consumed))
            if self._stop.wait(max(0,next_frame-time.monotonic())):
                break
            with self._condition:
                renderer, generation = self._renderer, self._generation
                state, version, status = self._state, self._version, dict(self._status)
                self._render_phase = "rendering"
            started = time.monotonic()
            try:
                jpeg = renderer(state)
                if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                    raise ValueError("Renderer must return a complete JPEG image")
            except Exception as exc:
                with self._condition:
                    if generation == self._generation:
                        self._render_error = f"{type(exc).__name__}: {exc}"
                        self._render_phase = "error"
                        self._renderer = None
                continue
            with self._condition:
                if self._stop.is_set():
                    break
                if generation != self._generation:
                    continue
                self._sequence += 1
                self._frame = Frame(self._sequence,jpeg,status,time.monotonic(),(time.monotonic()-started)*1000)
                self._render_phase = "ready"
                consumed = version
                self._condition.notify_all()
            next_frame = time.monotonic()+1/self.fps

    def close(self):
        with self._condition:
            if self._stop.is_set():
                return
            self._stop.set()
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._http_thread.join(timeout=5)
        self._render_thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
