"""Loopback MJPEG viewer with a single latest-state slot (no frame backlog)."""
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PAGE = rb"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>RTC simulation</title>
<style>
body{font:16px system-ui;background:#101827;color:#e6edf7;max-width:900px;margin:32px auto;padding:0 20px}
h1{font-size:26px}img{display:block;width:min(100%,640px);aspect-ratio:1;background:#202c40;border-radius:12px}
#status{padding:16px 0;line-height:1.7;white-space:pre-wrap}p{color:#aebbd0}
</style>
<h1>RTC simulation</h1>
<p>Live Kinetix view. Frames may be skipped; simulation time can differ from wall time.</p>
<img src="/stream.mjpg" alt="Waiting for a rendered simulation frame">
<div id="status" role="status">Connecting...</div>
<p>Viewing adds rendering overhead. Disable the viewer for timing measurements.</p>
<script>
async function refresh(){
 try{
  const response=await fetch('/status.json',{cache:'no-store'});
  if(!response.ok)throw Error('HTTP '+response.status);
  const s=await response.json();
  let lines=[s.phase];
  if(s.method)lines.push('Method: '+s.method+' | Predictor: '+s.predictor);
  if(s.episode!==undefined)lines.push('Episode: '+s.episode+' / '+s.episodes+' | Tick: '+s.tick);
  if(s.simulation_ms!==undefined)lines.push('Simulation time: '+(s.simulation_ms/1000).toFixed(2)+' s | Success: '+s.success);
  if(s.predicted_ms!==undefined){
   const actual=s.actual_ms==null?'awaiting response':s.actual_ms.toFixed(2)+' ms';
   lines.push('Predicted delay: '+s.predicted_ms.toFixed(2)+' ms | '+(s.delay_kind==='measured'?'Measured RTT: ':'Injected delay: ')+actual);
  }
  if(s.viewer_error)lines.push('Rendering error: '+s.viewer_error);
  if(s.error)lines.push(s.error);
  document.querySelector('#status').textContent=lines.join('\n');
 }catch(e){document.querySelector('#status').textContent='Viewer disconnected. Check the server and SSH tunnel.';}
 setTimeout(refresh,750);
}
refresh();
</script></html>"""


class SimulationViewer:
    """Render immutable simulator states off the controller thread.

    The renderer returns JPEG bytes. Only the latest state and JPEG are retained;
    browser writes never hold the producer lock. GPU rendering can still compete
    with inference for resources, so this mode is for functional inspection.
    """

    def __init__(self, port=8080, fps=10):
        if not math.isfinite(fps) or not 0 < fps <= 60:
            raise ValueError("Viewer fps must be in (0,60]")
        self.fps = fps
        self._condition = threading.Condition()
        self._closed = False
        self._renderer = None
        self._state = None
        self._sequence = 0
        self._jpeg = None
        self._frame_sequence = 0
        self._status = {"phase": "Loading policy / compiling; the first frame may take a while."}
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/stream.mjpg":
                    self.connection.settimeout(3)
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        last = -1
                        while True:
                            with viewer._condition:
                                viewer._condition.wait_for(lambda: viewer._closed or (
                                    viewer._jpeg is not None and viewer._frame_sequence != last), timeout=1)
                                if viewer._closed:
                                    return
                                if viewer._jpeg is None or viewer._frame_sequence == last:
                                    continue
                                jpeg, last = viewer._jpeg, viewer._frame_sequence
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                    except OSError:
                        pass  # A closed/slow browser must not block the simulation.
                    return
                if self.path == "/":
                    body, content_type = PAGE, "text/html; charset=utf-8"
                elif self.path == "/status.json":
                    body = json.dumps(viewer.status()).encode()
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self._http_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._http_thread.start()
        self._render_thread.start()

    @property
    def port(self):
        return self._server.server_address[1]

    def set_renderer(self, renderer):
        with self._condition:
            self._renderer = renderer
            self._condition.notify_all()

    def update(self, state=None, **status):
        with self._condition:
            self._status.update(status)
            if state is not None:
                self._state = state
                self._sequence += 1
            self._condition.notify_all()

    def status(self):
        with self._condition:
            return dict(self._status)

    def _render_loop(self):
        last, next_frame = 0, 0.
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or (
                    self._renderer is not None and self._sequence != last))
                if self._closed:
                    return
                remaining = next_frame - time.monotonic()
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                state, last, renderer = self._state, self._sequence, self._renderer
            try:
                jpeg = renderer(state)
            except Exception as exc:
                self.update(viewer_error=f"{type(exc).__name__}: {exc}")
                return
            with self._condition:
                self._jpeg = jpeg
                self._frame_sequence += 1
                self._condition.notify_all()
            next_frame = time.monotonic() + 1 / self.fps

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._http_thread.join(timeout=5)
        self._render_thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
