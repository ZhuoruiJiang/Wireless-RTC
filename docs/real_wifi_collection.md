# Collect real moving-laptop traces

All experiments in this plan use real network delays. Start the server with
`--delay none` and the probe client with `--delay none`. Walking changes the radio
environment naturally; no injected waits or packet impairment are needed.

## What the October 28 run establishes

Run: `runs/multitask-20261028-053121-038812Z`. The solve-rate difference is small:
RTC averages 0.91 versus wireless RTC at 0.89 (recent max) or 0.88 (rolling p95).
This single run does not establish a statistically reliable ranking or that
prediction is the only cause. The requests do establish an overprediction problem:

| Wireless predictor | Mean RTT (ms) | Mean prediction (ms) | RTT MAE (ms) | Excess delay steps (%) | Too few delay steps (%) |
|---|---:|---:|---:|---:|---:|
| Recent max | 14.93 | 39.92 | 27.31 | 54.97 | 2.70 |
| Rolling p95 | 14.79 | 26.80 | 15.38 | 46.28 | 4.23 |

Only 4.64% and 4.63% of their measured RTTs exceed the 20.00 ms control period.
Both inference-only RTC and a well-calibrated RTT predictor will often choose the
same single delay step in this setting. Extra network prediction need not improve
control when it does not change the required step count.

The 32-request maximum retains isolated spikes; the p95 also predicts an upper
tail rather than a central value. Wireless starts each episode at 60.00 ms; RTC
starts with measured warmed inference time. Excluding first requests still leaves
wireless MAEs of 26.32 ms and 13.78 ms. Cold starts are only part of the issue.
`prediction_mae_ms` uses different targets for RTC and wireless RTC; use
`rtt_prediction_mae_ms` for a common response-time target.

Chronological replay of BOTH wireless conditions' saved RTT sequences gives:

| Candidate | MAE (ms) | Excess delay steps (%) | Too few delay steps (%) |
|---|---:|---:|---:|
| Recent max, window 32 | 27.14 | 54.26 | 2.75 |
| p95, window 32 | 15.83 | 47.15 | 4.19 |
| Last value | 6.97 | 9.35 | 4.41 |
| Median, window 8 | 4.95 | 5.35 | 4.46 |
| p80, window 8 | 5.75 | 8.33 | 4.46 |
| p90, window 8 | 10.10 | 33.56 | 4.31 |

Replay resets predictor state at every episode, starts at 60.00 ms, predicts before
observing the next RTT, includes cold starts, and weights requests equally. These
are development diagnostics on historical data, not held-out results or new solve
rates: changing predictions can change actions, observations, request sizes and
timing. Validate on new sessions. `configs/rtc_real_candidates.json` provides a
fresh-run comparison of the original p95 with the short-window median and p80.
The existing user configuration is unchanged.

Reproduce this replay table (choose a new output filename each time):

```sh
uv run --no-sync python scripts/compare_saved_rtt.py --run runs/multitask-20261028-053121-038812Z --output runs/multitask-20261028-053121-038812Z/predictor_diagnostics.csv
```

## Collect a first session

From the repository root on the server:

```sh
uv run --no-sync python -m wireless_rtc serve --host 127.0.0.1 --port 8765 --delay none
```

Keep the same SSH route used for policy experiments. In a separate laptop terminal:

```sh
ssh -N -L 8765:127.0.0.1:8765 zhuorui@cybertron5.csres.utexas.edu
```

On the Windows laptop, start this and then walk your route:

```sh
uv run --no-sync python -m wireless_rtc probe --host 127.0.0.1 --port 8765 --samples 6000 --interval-ms 80 --payload-bytes 1024 --delay none --wifi --wifi-interface "Wi-Fi" --tag walk-route-a
```

The 80 ms interval is an inter-request pause AFTER completion, so actual request
rate also depends on RTT. Match the measured policy submission cadence rather than
treating this as an exact sampling frequency. Choose payloads close to actual
serialized policy traffic; the probe echoes the same payload in both directions,
which cannot match unequal observation/action sizes exactly.

Outputs:

- `trace.csv`: request outcome, full-precision RTT, server time, hold time, request
  monotonic start/end, and the most recently completed Wi-Fi sample BEFORE the
  request. RTT excludes connection establishment; the start/end envelope includes
  reconnect work when required. Errors remain explicit, with missing RTT.
- `wifi.csv`: timestamped background Wi-Fi polls, including errors/disconnections.
- `manifest.json`: destination, payload/cadence settings, and telemetry settings.

Wi-Fi defaults to a 1000 ms polling interval. Change it with `--wifi-interval-ms`;
the minimum is 100 ms. Polling uses cached `netsh wlan show interfaces` data and
does not initiate a scan. Driver updates may be slower than polling. Poll overhead
can still perturb client scheduling: compare a short stationary session with and
without telemetry. `wifi_age_ms` measures age since poll completion, not the age
of the driver's underlying observation.

## RSSI and SNR

On this laptop, the current English Windows output exposes actual `Rssi` in dBm,
signal percentage, AP BSSID, band, channel, radio type, and transmit/receive link
rates. The collector records those values when supplied. It never converts signal
percentage into a claimed RSSI measurement. Link rates are negotiated PHY rates,
not measured application throughput. For multiple adapters, select the adapter
used by the server route. Ethernet/VPN routing can otherwise bypass the Wi-Fi hop.

SNR requires both received signal and a measured noise floor:
`SNR_dB = RSSI_dBm - noise_dBm`. Standard netsh output here has no noise measurement,
so `wifi_noise_dbm` and `wifi_snr_db` stay blank. Never assume a constant noise
floor or encode missing SNR as zero. RSSI alone is enough to start testing whether
link features add predictive value. Windows permissions, localized output or
unsupported adapters produce explicit telemetry errors; RTT probing continues.

Linux `iw dev wlan0 link` / `iw dev wlan0 station dump` expose signal and available
station statistics; `iw dev wlan0 survey dump` may expose driver-dependent noise
and channel occupancy. Only compute SNR when meaningful signal and noise samples
are available on the same channel with compatible timestamps. Linux telemetry is
not implemented by this Windows collector. AP-side telemetry can add uplink RSSI,
retries and airtime if you have access; it is optional for initial data collection.

## Sessions and labels

Start with stationary strong coverage, stationary weak coverage, and repeated
walking routes through strong/weak coverage and AP transitions. Include stops and
return trips. Repeat at different times/days under natural traffic. Keep a small
session note with route, approximate segment times, laptop/adapter, power mode,
server/tunnel configuration, and whether other traffic was active. Keep the server
workload consistent across comparisons and record load when possible.

A practical starting dataset is several 10–15 minute sessions per condition
across multiple days. Treat entire sessions/routes/days as split units, with
separate training, validation and test sessions. Sample count alone does not
replace independent conditions. Never randomly shuffle nearby requests into
different splits. Near-duplicate overlapping history windows would leak context.

Only the laptop and the existing server endpoint are needed for the first stage;
no robot or AP modification is needed. But an echo probe only labels application
echo RTT. It does not include policy inference or reproduce its workload. After
probes, collect actual `rtc-live` request traces on the same routes before claiming
control improvements. The current `--wifi` flag is for `probe`; policy-run Wi-Fi
logging would need equivalent timestamp/sampler integration. Do not mistake a
loopback collector smoke check for training data or a campus Wi-Fi experiment.

Use total policy response time as the eventual control prediction label. Log
server inference/server duration separately to diagnose compute and transport
variation. RTT minus server duration is an aggregate residual including transport,
serialization and scheduling; it is not pure wireless delay. Clock synchronization
is unnecessary for a laptop-measured RTT label. True one-way labels require clock
synchronization or a different measurement design.

## Train and evaluate

Start with last value and short-window median/p80/p95 baselines, with zero added
delay. Choose window and quantile on validation sessions, then freeze them. Next,
try a small quantile boosting model using completed RTT lags, rolling median/tail/
variation, time since previous response, available RSSI level/trend, PHY rates,
AP transition flag, sample age/missingness, and known request/response size or task.
Current request RTT and current inference duration are targets/diagnostics, never
features. Join telemetry by availability timestamp before submission, not the
nearest sample (which could come from the future). AP identifiers should primarily
mark transitions; test on unseen routes/APs to check for location memorization.

Compare RTT-history-only against RTT-history-plus-link-features on the same held-out
requests. SNR/RSSI may add little when server load or upstream congestion dominates.
Predict median and upper quantiles; pick the control budget using validation
solve rate and under/over-prefix cost, rather than assuming p95 is universally best.
Report MAE, signed bias, quantile coverage/pinball loss, step mismatch, prediction
clipping, stale responses, buffer exhaustion, and final solve rate. Keep timeouts
as censored observations and report them separately; do not turn them into zero
delay or silently discard them. Existing `evaluate-trace` rejects error-containing
traces so that this issue cannot be hidden by success-only scoring.

Run paired episode seeds across methods and repeat independent sessions with
rotated condition order. Report confidence intervals for solve-rate differences,
respecting task/session grouping. Natural network states cannot be reset by a seed.
Avoid concurrent probe traffic against the single-client serial policy server.

CSV report cells now use two decimal places, including counts. Raw request/trace
logs and JSON summaries retain precision for subsequent aggregation and training.

References:

- [Microsoft: WLAN query interfaces and RSSI](https://learn.microsoft.com/en-us/windows/win32/api/wlanapi/nf-wlanapi-wlanqueryinterface)
- [Microsoft: driver-reported RSSI in dBm](https://learn.microsoft.com/en-us/windows/win32/api/wlanapi/ns-wlanapi-wlan_bss_entry)
- [Linux Wireless: iw measurements](https://wireless.docs.kernel.org/en/latest/en/users/documentation/iw.html)
- [scikit-learn: quantile gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
