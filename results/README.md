# Initial results

These are functional smoke tests from this Windows laptop on CPU. No Cybertron
or campus WiFi experiment has been run. No predictor configuration was tuned.

## Pretrained Kinetix RTC

Official epoch-24 grasp_easy checkpoint; five episodes per condition, seed 42,
eight-action chunks, 20 ms declared controller interval, minimum execution horizon
four steps. This is offline logical-time delay injection; wall-clock inference
speed does not determine the injected delay.

| Actual delay | Controller | Predictor | Solved | Buffer exhaustion |
|---|---|---|---|---|
| Fixed 40 ms | RTC | Recent max | 5/5 | 0 steps |
| Fixed 40 ms | Naive async | Recent max | 5/5 | 0 steps |
| Uniform 20-80 ms | RTC | Recent max | 5/5 | 0 steps |
| Uniform 20-80 ms | RTC | Rolling p95 | 5/5 | 0 steps |
| Uniform 20-80 ms | Naive async | Recent max | 5/5 | 0 steps |
| Uniform 20-80 ms | Naive async | Rolling p95 | 5/5 | 0 steps |

See `rtc-fixed-validated/` and `rtc-random/` for per-request and episode logs.
This confirms the controller runs under both delay sources. It does not establish
an RTC advantage: the baseline also succeeds, and the sample is very small.
The naive controller does not use its predictor; duplicated predictor conditions
therefore serve as a consistency check.

## Delay prediction

Synthetic traces have 1,000 observations; the first 32 are warmup. Every predictor
sees the same source trace, predicts before updating, and uses a rolling window of
32 (last-value uses one). Pinball loss uses q=.95 for all methods.

| Source | Predictor | MAE (ms) | Delay exceeds estimate | Mean prefix (steps) | Pinball loss (ms) |
|---|---|---:|---:|---:|---:|
| Fixed 60 ms | Recent max | 0.00 | 0.00% | 3.00 | 0.00 |
| Fixed 60 ms | Rolling p95 | 0.00 | 0.00% | 3.00 | 0.00 |
| Uniform 20-80 ms | Recent max | 27.82 | 2.69% | 4.00 | 1.44 |
| Uniform 20-80 ms | Rolling p95 | 25.40 | 7.85% | 4.00 | 1.48 |
| Uniform 20-80 ms | Last value | 19.74 | 48.97% | 3.02 | 9.87 |
| Bursty | Recent max | 41.12 | 4.86% | 4.80 | 2.76 |
| Bursty | Rolling p95 | 34.52 | 8.99% | 4.46 | 3.43 |
| Bursty | Last value | 8.61 | 49.69% | 2.63 | 4.30 |

Lower MAE alone does not imply a better scheduling predictor. Rolling p95 is less
conservative, but misses more upper-tail delays. On the IID trace it does not
reduce the rounded action prefix compared with the recent maximum.
Bursty traces have values outside the initial eight-action controller's available
overlap; this prediction-only study deliberately retains those values without
clipping them. These rows are not robot results.

See `initial-prediction/comparison.csv`, per-request CSVs, SVG plots, and manifests.

## Transport check

`loopback-random/` records 100 actual TCP request/response measurements with
artificial server waiting drawn uniformly from 10-70 ms.
`loopback-evaluation/` scores the final 68 measurements after warmup.
This validates the measurement path, not wireless performance.

The runs preceded the initial implementation commit, so their manifests record a
null original git revision. The matching implementation is committed in this repo;
retain this note rather than rewriting measurement-time provenance.
