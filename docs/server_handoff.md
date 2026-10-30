# Server handoff: Wireless RTC and Mininet-WiFi

## Status at branch creation (2026-10-30)

The user is moving toward Mininet-WiFi because live campus Wi-Fi comparisons are difficult to reproduce and mobile trials can break the SSH tunnel. This branch first preserves the existing experiment implementation and artifacts. **No Mininet-WiFi topology, installation, or emulated rollout has been implemented or verified yet.**

The intended deployment is one Linux GPU server running both the robot simulation/client and policy server, with a genuine TCP path through a Mininet-WiFi station, access point, and server-side host. The logical client/server separation remains necessary to exercise network latency, retransmissions, queues, and RTC timing. Do not connect client and policy through localhost in a way that bypasses the emulated link. SSH is for management, not the simulated experimental link.

Before implementation, inspect the server's Linux kernel, administrative access, network namespaces, mac80211_hwsim, wmediumd, and Mininet-WiFi support. Availability on the shared GPU server is unconfirmed. Avoid disrupting its physical network. Keep CPU-side client work from interfering with GPU inference measurements. Use fixed scenario configurations and seeds, but do not claim real-time execution is bitwise deterministic. Record topology, propagation/interference models, workload, versions, and seeds with each run. Preserve real-network data for validation and label simulated results separately.

## Existing experiment

- Persistent TCP application protocol with newline-delimited JSON; current real deployment forwards traffic through SSH to a GPU policy server. Full response duration includes communication and processing; inference duration is separate.
- Current rollout tasks: `catcher_v3`, `hard_lunar_lander`, `mjc_walker`. Older strong-condition runs include a different task set and must not be pooled with these trials.
- Main heuristic predictors: last response, recent maximum, rolling p95, using history window 8. Rolling p50/p70/p80/p90 are internal comparisons for the underprediction/task-success hypothesis. All rate outputs named `_pct` are percentages.
- Existing `qgb_rtt` checkpoint is a frozen internal reference; no more QGB development. Fugu-inspired MLP work is deferred until heuristic experiments are complete.
- Current control interval is 20 ms and chunk horizon is 8. Use saved configurations rather than assuming these defaults apply to every historical run.
- Actual response arrival controls which action indices are valid. Predictions affect guidance, not a pause in physics. Current fallback is zero action. Underprediction, buffer deadline misses, expired responses, fallback ticks, and task failure are distinct events.
- Same rollout episodes provide task success and controller behavior metrics. Retain task-level results and denominators; do not pool incompatible action spaces for action-jump metrics.

## Explicit run selection

`configs/network_summary.json` selects two strong stationary dorm traces and the weak stationary hallway trace `network-characterize-20261030-064156-590640Z`. Historical outdoor runs, including `015059`, are also weak stationary runs, but are not selected for the current comparison. Legacy folder names such as `stable` or `walking` are not authoritative reporting labels.

`configs/midterm_prediction.json` references the network selection configuration. It does not need to discover the latest Table 1 report. `configs/rollout_summary.json` explicitly lists rollout blocks. At branch creation, its strong selection contains only block 1; weak contains four blocks. Update the selection explicitly when additional compatible completed strong blocks are ready. Strict completeness checks may reject partial selections; do not weaken those checks merely to produce a table.

```bash
uv run --no-sync python -m wireless_rtc network-summary --config configs/network_summary.json
uv run --no-sync python -m wireless_rtc prediction-summary --config configs/midterm_prediction.json
uv run --no-sync python -m wireless_rtc rollout-summary --config configs/rollout_summary.json
```

See `docs/result_selection.md`, `docs/network_characterization.md`, and `docs/robot_rollouts.md` for details. Selection snapshots record resolved runs and configuration hashes. Source manifests and raw CSVs must remain intact. Historical manifests can contain Windows paths; use repository-relative selection paths for portability and verify that any replay/checkpoint path resolves on Linux.

## Environment and validation

The previous remote policy host was `cybertron5.csres.utexas.edu`. Pretrained policy weights are expected under `checkpoints/24/policies` on that server; those weights were absent from the local checkout. A typical server invocation before emulation is:

```bash
uv run --no-sync python -m wireless_rtc serve --host 127.0.0.1 --port 8765 --checkpoint-dir checkpoints/24/policies --delay none
```

This localhost binding is for the previous SSH-forwarded setup; adapt binding and routing for emulation. Do not assume the existing environment's Python/GPU dependencies are already installed in a fresh checkout. Inspect the README and dependency configuration first.

At branch creation, `uv run --no-sync python -m unittest discover -s tests -q` ran 107 tests successfully, with five optional tests skipped. This validates the existing experiment code, not a Mininet-WiFi integration.

## Context in a new Codex session

Git transfers these files, not the complete previous Codex conversation or uncommitted session state. Ask a fresh server session to read `AGENTS.md` and this document first. Keep this handoff updated when decisions or implementation status change. Follow `docs/artifact_tracking.md` to commit and push all experiment artifacts without committing credentials or dependency caches.
