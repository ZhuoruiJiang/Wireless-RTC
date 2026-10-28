# Response-delay and WiFi experiments

## Establish the limiting component

The wireless hop is a candidate source of latency variation, not an established
bottleneck. Numeric observations/action chunks have different traffic demands
from image streams. Compare wired Ethernet against stationary WiFi using the same
server, workload, request cadence, payloads, and SSH tunnel. Repeat across sessions.

Use request `inference_ms`, `server_ms`, and actual injection sleep alongside RTT.
RTT minus `server_ms` is a transport/serialization/dispatch/decoding residual, not
pure wireless delay. A local echo/probe baseline helps, but its default 1 KiB echo
is not the policy's observation/action payload. Server load and client deadline
overruns must be recorded alongside link conditions.

## Test in stages

1. **Repeatable application stress.** Baseline plus seeded normal waits of 60/20,
   100/30 and 150/50 ms (mean/std). Use matched episode seeds and multiple delay
   seeds. Measure solve rate, buffer exhaustion, expired responses, control-budget
   overruns, prediction exceedance, and excess predicted prefix length.
2. **Packet-level emulation where available.** Linux netem can emulate packet
   delay distributions, loss, reordering, and rate limits. It requires network
   administration capabilities; use an isolated lab machine/VM or administrator
   assistance. Application sleeping requires neither root nor host network changes.
   An added wait per response is not the same as adding delay to each network hop.
3. **Real measurements.** Collect stationary good-signal, weaker-signal, busy-hour,
   and controlled competing-traffic sessions. Walking a repeatable route is useful
   specifically for roaming/handoffs; stationary conditions should come first.
   Record timestamp, location category, SSID/BSSID, signal/link rate where available,
   and whether the laptop changed access points. Use actual policy-request traces.
   Controlled traffic should use equipment/network access you are permitted to test.

Collect traces once and compare predictors chronologically on the same held-out
trace, respecting session/episode resets. Do not randomly shuffle requests. Tune
on development sessions, freeze choices, then evaluate on different sessions.
Replay can reuse measured delays in the offline scheduler; it does not recreate
the original radio conditions. Never report synthetic stress as measured WiFi.

## Predictor progression

Keep last value, recent max and rolling p95 as baselines. First try a rolling
quantile with a modest tuned safety margin, an exponentially weighted location/
variation estimate, and a fast-increase/slow-decrease upper estimate. Assess tail
coverage and excess prefix cost together; MAE alone favors central forecasts.

For independent stationary normal draws, past draws contain no information about
the next random deviation. The history can estimate distribution parameters; an
ideal 95th-percentile prediction is the clipped `mu + 1.645*sigma`. This is an
oracle diagnostic, not a deployable history-only prediction. It may exceed the
160 ms action horizon, where better prediction cannot prevent chunk expiration.

After collecting real correlated traces, try a small conditional-quantile model
(for example quantile gradient boosting), using prior RTTs, recent quantiles and
variation, trend, payload size, and available link/server telemetry. Every feature
must be available before request submission; current inference time, injected
delay, and future RTT are not valid inputs. Start small before adding recurrent
neural models. Compare against the strongest heuristic on held-out sessions.

References:

- [Linux netem manual](https://man7.org/linux/man-pages/man8/tc-netem.8.html)
- [Cisco Meraki: wireless throughput testing](https://documentation.meraki.com/Wireless/Troubleshooting_and_Support/Troubleshooting/Measure_Wireless_Throughput_Using_iPerf)
- [RFC 6298: smoothed RTT and variation](https://www.rfc-editor.org/rfc/rfc6298.html)
- [scikit-learn: quantile gradient boosting](https://sklearn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
