# First experiment

Question: does a causal delay estimate preserve RTC performance under variable response delays?
No learned predictor or telemetry feature tuning is part of this stage.

1. Check prediction on fixed, IID uniform, and correlated bursty synthetic traces.
2. Compare plain policy, inference-only RTC, and network-aware RTC on grasp_easy, using fixed
   and IID random injected delays. Five episodes are a smoke test, not a success-rate
   estimate suitable for drawing scientific conclusions.
3. Measure real RTT to Cybertron with the probe service, repeat across WiFi sessions.
4. Evaluate the same predictors chronologically on those held-out traces.
5. Run the paced live simulator client (`rtc-live`) and compare with replay before
   claiming live wireless robot-control performance.

## Timing contract

An observation and aligned previous chunk refer to request tick t.
Predicted d controls guidance only. Actual delay controls action availability.
Offline runs use ceil(ms/control_ms); exact multiples arrive at that tick.
The paper idealizes sub-tick timing; this code explicitly adopts conservative rounding.
We retain unbounded predictions in logs and record whenever the applied prefix is clipped
to the valid overlap. Never reinterpret a clipped prediction as an accurate prediction.

The scheduler has one request in flight. Next inference starts after
s = max(s_min, actual_delay_steps) ticks. During those ticks execute old actions before
arrival, then the new chunk at its elapsed indices. Thus s is a realized quantity;
we do not use future actual delay when generating the chunk.
A zero-action fallback fills genuinely missing buffer coverage. Physics never pauses.
This is not a safe real-robot fallback and is not intended for hardware use.

Offline logical-time delays include the WHOLE modeled response budget; actual host
inference time is measured separately and is not added a second time.
The initial action chunk is generated before the episode controller begins.

## Live RPC timing contract

`rtc-live` executes local physics while one background policy RPC is in flight.
It uses actual measured response delay, with no synthetic delay injection. Requests
carry the observation tick and a previous chunk aligned to that tick. Prefix
guidance is based only on completed past responses. A chunk received at simulation
tick `r` for observation tick `t` is used from index `r-t` onwards, never index zero.
Chunks with `r-t >= H` are discarded. Buffer exhaustion advances physics with zero
actions. Successful expired responses still contribute to delay history. Responses
drained after an episode ends are logged but never applied or fed to its predictor.

The clock starts after physics/policy warmup and an initial plain chunk. Server RNG
and predictor history reset each episode. In-flight requests are drained before
starting a new episode. All three modes use the same controller. `naive` uses
plain inference without a predictor. `rtc` and `wireless_rtc` use guided inference,
with feedback from past server inference timings and total RTTs, respectively.
A warmed guided-inference measurement initializes `rtc` before the episode and
is recorded in episode logs. Real network conditions differ between runs.

New manifests map mode names to predictor inputs. Results from before the
three-mode protocol used `rtc` for what is now called `wireless_rtc`. Compare
labels across versions only after examining manifests and software revisions.

Offline `rtc` uses host inference duration as predictor feedback; `wireless_rtc`
uses injected whole-response delay. Host compute time is not added to the logical
delay. Logs distinguish the predictor target from the actual response delay.
`naive` is run once per comparison, without meaningful prediction scores.

RTT is measured from request submission through response validation on the client.
It includes local worker dispatch/serialization, transport, policy inference and
decoding. It excludes SSH/TCP connection setup and episode initialization. Responses
are consumed at the next available controller boundary. `ideal_delay_steps` derives
from wall-clock RTT, but alignment uses `elapsed_ticks` at actual application; these
can differ if local physics overruns the control budget. Controller-work overruns
and buffer exhaustion are recorded per tick. The controller does not accelerate
physics to catch up after missed deadlines. Viewer rendering adds resource contention
and must be disabled for timing measurements. A timeout is censored and stops the
experiment with error status; it is never substituted with a completed delay.

## Prediction scoring

Predict first, then observe the completed request. Initial estimates are explicit.
Exclude 32 warmup samples from summary metrics but retain them in raw logs.
Report MAE, signed bias, continuous and tick-level exceedance rates, mean predicted
prefix, mean overestimation, and common-q pinball loss (q=.95 for every predictor).
An empirical rolling p95 is not guaranteed to achieve 95% coverage out of sample.
Prediction exceedance is NOT buffer exhaustion and NOT task failure.
MAE favors central forecasts; scheduling needs tail coverage too.

All predictors get identical exogenous synthetic delays. Policy RNG and delay RNG are
independent. Synthetic burst parameters are illustrative, not a fitted campus WiFi model.
Raw traces and resolved configs are saved. Existing output folders are never overwritten.

## Later configuration experiments

Keep development and final evaluation sessions separate. Tune on development sessions,
freeze configuration, then evaluate on untouched sessions. Split by session/time, not
random rows. Compare history-only against added WiFi telemetry to establish its value.
Do not tune to the included seed-42 demonstration and call it a held-out result.
