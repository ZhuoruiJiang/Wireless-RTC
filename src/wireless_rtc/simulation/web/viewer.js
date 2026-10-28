"use strict";
(() => {
  const element = id => document.getElementById(id);
  const canvas = element("scene");
  const context = canvas.getContext("2d");
  let session = "", sequence = 0, paused = false, timer, inFlight = false, failures = 0;
  let lastPaint = 0, fps = 10, generation = 0;
  const text = (id, value) => { element(id).textContent = value; };
  const number = value => typeof value === "number" && Number.isFinite(value);
  const milliseconds = value => number(value) ? `${value.toFixed(1)} ms` : "—";

  function badge(label, mode) {
    text("connection", label);
    element("connection").dataset.mode = mode;
  }
  function notice(message) {
    element("notice").hidden = !message;
    text("notice", message || "");
  }
  function progress(snapshot) {
    const state = snapshot.state;
    fps = Math.max(1, Math.min(60, snapshot.fps || 10));
    text("phase", state.phase || "Waiting for simulation");
    text("method", state.method || "—");
    text("predictor", state.predictor || "—");
    text("episode", state.episode == null ? "—" : `${state.episode} / ${state.episodes || "?"}`);
    text("tick", state.tick ?? "—");
    text("sim-time", number(state.simulation_ms) ? `${(state.simulation_ms / 1000).toFixed(2)} s` : "—");
    text("success", state.success == null ? "—" : state.success ? "Yes" : "No");
    text("prediction", milliseconds(state.predicted_ms));
    text("actual", milliseconds(state.actual_ms));
    element("injection-row").hidden = !number(state.injected_delay_ms);
    text("injection", milliseconds(state.injected_delay_ms));
    element("exhaustion-row").hidden = !number(state.buffer_exhaustion_steps);
    text("exhaustion", state.buffer_exhaustion_steps ?? "—");
    text("delay-label", state.delay_kind === "measured" ? "Measured RTT" : "Injected delay");
    const error = state.error || snapshot.renderer.error;
    notice(error);
    if (error) badge("Needs attention", "error");
    else if (/completed/i.test(state.phase || "")) badge("Completed", "completed");
    else if (paused) badge("View paused", "paused");
    else if (!lastPaint) badge("Warming up", "warming");
    else if (snapshot.renderer.age_ms > 3000) badge("Waiting for frames", "warming");
    else badge("Live", "live");
    text("waiting", error || state.phase || "Waiting for the first rendered frame…");
  }

  async function paint(frame, requestGeneration) {
    const bytes = Uint8Array.from(atob(frame.jpeg), ch => ch.charCodeAt(0));
    let bitmap;
    try {
      bitmap = await createImageBitmap(new Blob([bytes], {type: "image/jpeg"}));
    } catch (cause) {
      const error = Error(`Could not decode simulation frame: ${cause.message}`);
      error.name = "FrameDecodeError";
      throw error;
    }
    try {
      if (requestGeneration !== generation || paused) return;
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      context.drawImage(bitmap, 0, 0);
      canvas.classList.add("ready");
      element("placeholder").hidden = true;
      sequence = frame.sequence;
      lastPaint = performance.now();
      const tick = frame.state.tick == null ? "" : ` · tick ${frame.state.tick}`;
      text("frame-info", `Frame ${sequence}${tick}`);
    } finally {
      bitmap.close();
    }
  }

  async function poll() {
    if (inFlight) return;
    inFlight = true;
    const requestGeneration = generation;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    try {
      const query = new URLSearchParams({after: String(paused ? Number.MAX_SAFE_INTEGER : sequence), session});
      const response = await fetch(`/api/snapshot?${query}`, {cache: "no-store", signal: controller.signal});
      if (!response.ok) throw Error(`Viewer returned HTTP ${response.status}`);
      const snapshot = await response.json();
      if (requestGeneration !== generation) return;
      if (session !== snapshot.session) {
        if (session) {
          lastPaint = 0;
          text("frame-info", "New run connected; waiting for its first frame");
        }
        session = snapshot.session;
        sequence = 0;
      }
      if (snapshot.frame && !paused) await paint(snapshot.frame, requestGeneration);
      failures = 0;
      progress(snapshot);
    } catch (error) {
      if (requestGeneration === generation) {
        failures += 1;
        const decodeError = error.name === "FrameDecodeError";
        badge(decodeError ? "Image error" : "Disconnected", decodeError ? "error" : "disconnected");
        notice(decodeError ? error.message : `Cannot load the local viewer. Retrying automatically. ${error.message}`);
        text("waiting", "Waiting for the local viewer to reconnect…");
      }
    } finally {
      clearTimeout(timeout);
      inFlight = false;
      timer = setTimeout(poll, failures ? Math.min(3000, 300 * failures) : 1000 / fps);
    }
  }

  element("pause").addEventListener("click", () => {
    paused = !paused;
    text("pause", paused ? "Resume view" : "Pause view");
    badge(paused ? "View paused" : "Connecting", paused ? "paused" : "connecting");
    if (!paused) sequence = 0;
  });
  element("retry").addEventListener("click", () => {
    generation += 1;
    session = "";
    sequence = 0;
    failures = 0;
    clearTimeout(timer);
    badge("Connecting", "connecting");
    if (!inFlight) poll();
  });
  poll();
})();
