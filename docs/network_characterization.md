# Midterm network characterization (Table 1): strong and weak stationary Wi-Fi

Use the Desktop `Wireless-RTC` checkout. This stage measures a **fixed-observation
policy-response workload**, not a robot rollout or task success. It repeats a valid
reset observation for `cartpole_thrust`, uses guided inference with a one-step
prefix, and keeps the previous chunk and overlap fixed. Both settings therefore
use the same observation, task, checkpoint and payload shape. The client needs its
normal Kinetix simulator dependencies but no local policy checkpoint.

RTT includes serialization, communication, inference, decoding and action
validation. Connection setup, initial policy compilation, initialization and five
warmup requests are excluded. An 80 ms pause follows each measured response; this
is not a fixed 12.5 Hz sampling rate. The timer starts after the initial radio poll.
The last request may finish up to one timeout after the requested duration.

## 1. Server and tunnel

On Cybertron, use the normal policy server, with artificial delay disabled:

```bash
uv run --no-sync python -m wireless_rtc serve --host 127.0.0.1 --port 8765 --checkpoint-dir checkpoints/24/policies --delay none
```

The server protocol is unchanged; these client changes do not require updating
server code. Stop a previously running server before replacing it. In a laptop
terminal, keep the existing tunnel, or start it if needed:

```powershell
ssh -N -L 8765:127.0.0.1:8765 zhuorui@cybertron5.csres.utexas.edu
```

Do not run probes or controller clients concurrently against this serial policy
server. Keep the same server and tunnel for all sessions. Ensure the server route
uses Wi-Fi rather than Ethernet. No viewer is started by characterization.

## 2. Short development check

In another laptop terminal, from the repository root:

```powershell
uv run --no-sync python -m wireless_rtc network-characterize --setting strong --split development --duration-s 30 --wifi-interface "Wi-Fi" --notes "Dorm desk; development check"
```

The default experiment is `midterm`, default workload is `policy`, and default task
is `cartpole_thrust`. If your adapter has another name, replace `Wi-Fi`. Errors are
saved in `wifi.csv`; unavailable RSSI remains missing. Standard Windows output may
provide signal percentage without actual RSSI; no dBm conversion is invented.
Confirm that the completion summary has responses and a numeric median RSSI.
If Wi-Fi queries fail, inspect `wifi_error` (Windows location access may be needed).
The signal collector only queries cached interface information; it performs no scans.

## 3. Held-out sessions

The revised main comparison uses **two stationary locations**: strong coverage at
an indoor dorm desk and weak coverage at a fixed outdoor location. Mobility
attempts remain exploratory because the SSH tunnel can break. Do not pool those
attempts with the stationary weak location. AP, band, or interference can differ,
so the comparison does not isolate signal strength as a causal variable.

For the indoor strong-location run:

```powershell
uv run --no-sync python -m wireless_rtc network-characterize --setting strong --duration-s 600 --wifi-interface "Wi-Fi" --notes "Dorm desk; stationary strong Wi-Fi"
```

For a stationary outdoor weak-location run:

```powershell
uv run --no-sync python -m wireless_rtc network-characterize --setting weak --duration-s 600 --wifi-interface "Wi-Fi" --notes "Fixed outdoor student apartment location; stationary weak Wi-Fi"
```

One ten-minute run per location supports the initial pilot comparison; repeat both
if time permits. Keep laptop placement and orientation similar within a setting.
The existing completed outdoor session `network-characterize-20261030-020233-934624Z`
was moved from `walking/test` to `weak/test`; its raw logs are unchanged, and its
manifest preserves the original label and notes. The earlier seven-BSSID mobility
session remains under `walking/test` and is excluded by the settings filter below.
Collection ends automatically. No tags or manual session numbers are required.

## 4. Export Table 1

**Current selection interface:** edit `configs/network_summary.json` and run
`network-summary --config configs/network_summary.json`. Only listed runs are
included; no folder moves or manifest edits are needed. See
[the selection guide](result_selection.md). The legacy map described below applies
only to old direct-library workflows, not the current report command.

The report command reads `configs/network_summary.json`. Its `runs` lists are the
complete input selection: every unlisted session is ignored, even if its manifest
says test. A missing selected path is an error. You do not need to move folders or
edit saved split labels. `session_map.json` is not read by this command.

```powershell
uv run --no-sync python -m wireless_rtc network-summary --settings strong weak
```

This selects completed `test` sessions of the `policy` workload only for `strong`
and `weak`, pools successful
responses within each setting, and generates `table1.csv`, `table1.tex` and an audit
`summary.json`. It prints the report folder and request-error counts. The export
does not modify the report source or copy unmeasured numbers into it.

RSSI is aggregated from independent in-window `wifi.csv` polls, not per-request
copies. Failed requests are excluded from successful-response percentiles and
counted explicitly. Interrupted/failed sessions are excluded and listed in the
audit. If only one setting exists, the output contains only that setting and the
audit names the missing setting. Missing RSSI is blank in CSV and a dash in LaTeX.
The exporter rejects incompatible task/checkpoint/workload/cadence combinations.
Changing experimental settings intentionally requires a new `--experiment` name.

## Table 2: offline predictor evaluation

**Current selection interface:** `configs/midterm_prediction.json` references
`configs/network_summary.json`. The configured runs are read directly; the command
does not automatically choose a latest Table 1 export. See
[the selection guide](result_selection.md). An explicit `--table1` can still select
an existing report intentionally.

To evaluate the configured sessions directly, run:

```powershell
uv run --no-sync python -m wireless_rtc prediction-summary
```

This uses the explicit runs referenced by `configs/midterm_prediction.json`; no
Table 1 export is needed and no latest-report lookup is performed. Each session starts a fresh eight-response history;
the first eight responses are excluded for every method. No network connection or
new collection is needed. It reports last response, recent maximum, and rolling p50/p70/p80/p90/p95
at a 20 ms control interval. No parameters are fitted or selected from these test
data. `configs/midterm_prediction.json` points to the existing trusted QGB checkpoint
for internal p95 comparison; set `model` to null in a separate config if evaluating
only heuristics. Model weights are not changed. A missing/incompatible model fails
explicitly instead of silently omitting it.

Results are saved under `runs/midterm/prediction-summary/reports/table2-<timestamp>/`:
`table2.csv` and `table2.tex` contain last response, recent maximum and rolling p95,
with a pending MLP row in LaTeX. `internal_metrics.csv` additionally retains rolling
p50/p70/p80/p90 and the QGB checkpoint; `predictions.csv`
retains unrounded per-request forecasts; `metrics.json` retains unrounded metrics.
The manifest records source hashes, the Table 1 selection, sample counts and warmup
counts. Failures, long gaps and changed workload settings require an explicit
scoring decision and are not silently filtered. All observed rates in internal CSV
and JSON metrics are percentages (0-100), with `_pct` suffixes: coverage, exceedance,
step underprediction and step overprediction. Fraction-valued duplicate rate
columns are omitted. Mean prefix/excess steps and MAE retain their original units.

## Output layout

```text
runs/
  midterm/
    network-characterization/
      strong/
        development/network-characterize-<UTC timestamp>/
        test/network-characterize-<UTC timestamp>/
      weak/
        test/network-characterize-<UTC timestamp>/
      reports/table1-<UTC timestamp>/
```

Each collection folder contains `manifest.json`, `trace.csv`, `wifi.csv`, and
`summary.json`. Startup failures may have only a manifest. Ctrl+C preserves a
partial run and marks it `interrupted`, so it cannot silently enter the final
table. Network errors during collection are retained and the client reconnects;
malformed policy responses stop the run and mark it failed.

`--experiment NAME` selects another group. All existing experiment commands also
support `--experiment`, `--setting`, and `--split` for similarly grouped new runs.
For example, later `rtc-live` runs can use
`--experiment midterm --setting strong --split test` and will be stored under
`runs/midterm/rtc-live/strong/test/`. Existing run folders and saved models are
not relocated, preserving checkpoint paths. Legacy commands without grouping
arguments keep their previous output behavior. `--output` still selects an exact
new directory; automatic table discovery expects the organized layout above.

For transport-only diagnostics, `--workload echo --no-wifi` works without simulator
dependencies. Echo results are a different target; the policy Table 1 exporter
never includes them. Export echo data explicitly with `network-summary --workload echo --settings strong weak`.
Loopback tests verify the collector; they are not wireless experiment results.
