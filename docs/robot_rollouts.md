# Midterm Kinetix rollouts: Tables 3 and 4

This stage runs **simulated Kinetix control tasks on the laptop** and sends policy
requests to the remote server over the existing SSH tunnel. No physical robot is
controlled. It uses actual response delays, without injected waits.

The main tasks are `catcher_v3`, `hard_lunar_lander`, and `mjc_walker`. Keep
the laptop stationary at the previously measured strong or weak location. Do not
run the characterization collector or another policy client at the same time.

## Server

Use the normal server with all three checkpoints available:

```bash
uv run --no-sync python -m wireless_rtc serve --host 127.0.0.1 --port 8765 --checkpoint-dir checkpoints/24/policies --delay none
```

Keep the existing SSH tunnel. The server protocol is unchanged by these client
logging additions. The midterm client rejects injected-delay servers. Startup and
policy warmup use the CLI's 120-second timeout; timed episode requests use the
config's 10-second timeout. Failures stop the block and preserve partial files.

## First: short development check

From the Desktop `Wireless-RTC` repository, run:

```powershell
uv run --no-sync python -m wireless_rtc rtc-live --config configs/midterm_rollout_smoke.json --experiment midterm --setting strong --split development
```

This runs one catcher episode for each of naive execution, inference-only RTC,
and wireless RTC with rolling p95. It checks simulator/server compatibility,
initialization, telemetry and timing. It does not establish a success rate.
Do not add `--view` to timing runs. Compilation can take time before the first
episode. Confirm the manifest finishes with `status: completed`; check tick timing
and Wi-Fi errors before beginning final measurements. A missing median signal in
the older characterization summaries does not become a fabricated signal here:
radio errors remain explicit in `wifi.csv` and per-request fields.

## Final measurements: four numbered blocks per setting

For a separate five-task diagnostic of inference-only RTC with recent maximum:

```powershell
uv run --no-sync python -m wireless_rtc rtc-live --config configs/rtc_recent_max_5tasks.json --experiment rtc-diagnostics --setting weak --split development
```

This runs only `rtc` + `recent_max` for 20 episodes on each of `grasp_easy`,
`cartpole_thrust`, `catcher_v3`, `hard_lunar_lander`, and `mjc_walker` (100 episodes).
The maximum uses eight completed server inference times, not full response times.
It runs as one batch without `--block`. Per-task `rtc-recent_max/episodes.csv`
identifies failed episode indices and seeds; `requests.csv` and `ticks.csv` retain
delay, expiration and buffer-exhaustion diagnostics. Root `task_summary.csv` reports
each task separately. The diagnostics experiment stays separate from midterm data.
For visual inspection, optionally add `--view --view-port 18083`; rendering adds
local load, so use runs without the viewer for timing comparisons.

`configs/midterm_rollout.json` defines 20 total episodes per task/condition, split
into **four blocks of five episodes**. There are nine conditions:

- Naive asynchronous execution once.
- Inference-only RTC with recent maximum of server inference durations.
- Wireless RTC with last response, recent maximum, and rolling p50/p70/p80/p90/p95.

The seven wireless predictors share an eight-response window and a 60 ms initial
estimate. Inference-only RTC is initialized with a warmed server inference timing.
Predictor state and policy RNG reset every episode. Episode seeds are paired across
methods and settings: block 1 uses 42--46, then 47--51, 52--56 and 57--61. Condition
order rotates with block/task. Mask construction and zero-action fallback remain
fixed. The report tables include only the agreed main conditions; p50--p90 remain
in internal results for sensitivity plots.

Start block 1 at the dorm desk:

```powershell
uv run --no-sync python -m wireless_rtc rtc-live --config configs/midterm_rollout.json --experiment midterm --setting strong --split test --block 1
```

At the fixed outdoor weak location, run:

```powershell
uv run --no-sync python -m wireless_rtc rtc-live --config configs/midterm_rollout.json --experiment midterm --setting weak --split test --block 1
```

Repeat both commands with `--block 2`, `--block 3`, and `--block 4`. Each command
runs 135 episodes (three tasks, nine conditions, five episodes each). All four
blocks total 540 episodes per setting. With the default 256-step task horizon,
nominal simulated episode duration is at most 5.12 seconds, but startup, warmup,
logging, draining and local overruns add wall time. Use the smoke run and first
block to assess actual runtime. You can pause between blocks. Alternate settings
between blocks if practical to reduce time-of-day confounding. Do not move during
a block labeled strong or weak stationary Wi-Fi.

Avoid duplicate completed block numbers within a setting. After an interruption,
rerun that same block; the new folder preserves the failed attempt. Choose the completed attempt in `configs/rollout_summary.json`. The exporter ignores
unlisted attempts, errors on selected incomplete attempts, and rejects duplicate
completed blocks rather than counting identical seeds twice. Session failures
must be discussed separately when interpreting results; a completed-block analysis
does not establish recovery from broken SSH connections.

## Outputs and summaries

**Current selection interface:** edit the exact block paths in
`configs/rollout_summary.json`. Only listed blocks are summarized. See
[the selection guide](result_selection.md) for repeats, partial sets, and filtering
one setting. Incompatible task sets are still rejected.

New block folders are organized as:

```text
runs/midterm/rtc-live/strong/test/block01-<UTC timestamp>/
runs/midterm/rtc-live/weak/test/block01-<UTC timestamp>/
```

Each block stores a manifest, `wifi.csv`, per-task/per-condition `requests.csv`,
`ticks.csv`, `episodes.csv`, plus existing success/latency summaries. Raw tick logs
include commanded action vectors, valid chunk-switch markers, and normalized
boundary action jumps. Actions are logged before the simulator's actuation noise;
these are commanded-action continuity metrics. Normalization uses the environment
action range. Fallback transitions are excluded from the valid-boundary jump.

Request logs include remaining overlap and the absolute tick at which that
coverage ends. A usable response consumed at that first uncovered tick is on time.
A fully expired chunk is unusable even if it arrives at that boundary. Responses
drained after episode termination are censored for deadline analysis unless buffer
coverage already ran out while the controller was executing. Local tick overruns,
clipping, failures and post-termination responses remain visible.

After all eight blocks finish:

```powershell
uv run --no-sync python -m wireless_rtc rollout-summary
```

For an early preview after one or more completed blocks:

```powershell
uv run --no-sync python -m wireless_rtc rollout-summary --allow-partial
```

Exports go under `runs/midterm/rollout-summary/reports/tables3-4-<timestamp>/`:

- `table3.csv` / `table3.tex`: equal-weight task success percentages for main methods.
- `table4.csv` / `table4.tex`: deadline misses, expired responses, fallback ticks,
  and the mean of per-task p95 normalized boundary jumps.
- `internal_rollout_metrics.csv`: all conditions, including p50--p90.
- `tradeoff.csv`: wireless conditions' success and prediction metrics from the
  **same rollout trials**, for later plots.
- `per_task_metrics.csv`: task-specific values, event counts and denominators.
- `summary.json`: full-precision metrics, included/excluded blocks and limitations.

The summary pools events within each task, then gives tasks equal weight. Prediction
diagnostics exclude the first eight consumed responses per episode; raw logs and
success rates include episode startup behavior. Short episodes may have no scored
prediction samples; their forecast diagnostics remain unavailable. Rates use
percentages and `_pct` column names. Missing measurements are blank, not zero.

Success uncertainty uses a reproducible percentile bootstrap of complete rollout
blocks, resampled jointly across tasks. These intervals are preliminary with only
four blocks; no interval is produced for one block. Do not treat the number of
requests as the number of independent task-success trials. This exporter does not
edit the LaTeX report automatically. Keep partial exports separate from final ones.
