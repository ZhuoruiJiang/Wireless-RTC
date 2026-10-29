# Offline quantile gradient boosting

The comparison script evaluates the full `data/commect/measured.csv` by default.
Run the heuristic comparison with
`uv run --no-sync python scripts/compare_rtt_segment.py --history 8 --tag full-dataset`.
Each session's first eight measurements supply history and are excluded from scores;
the excluded warmup count is recorded. Select just the unstable interval with
`--segment data/commect/unstable_segment.csv`.
Its default rolling window is eight completed measurements; change `--history`
to compare other windows. Include `qgb_rtt` without retraining by adding
`--model runs/YOUR-SAVED-RUN/model.pkl` (replace with an existing model path).
When including a model, match its trained history length. `qgb_rtt_snr` is excluded.
The saved-model overlap guard still applies: a full source that includes the
model's training/validation records cannot serve as its held-out test. Select
unseen target rows with `--segment`, or use `quantile-evaluate` on the test split.

This implements the first learned delay predictor, independently of the policy
controller. It uses scikit-learn's histogram gradient boosting with quantile loss.
Separate models estimate median, p80 and p95 RTT. A later MDN can use the same CSV
schema, raw-record partitions, history builder and evaluation metrics.

No artificial network delays are introduced. Test fixtures check correctness;
they are not measured datasets or evidence of prediction performance.

## Install and run

Run from the updated Desktop `Wireless-RTC` checkout:

```powershell
uv pip install "scikit-learn>=1.5,<2"
uv run --no-sync python scripts/prepare_commect.py
uv run --no-sync python -m wireless_rtc quantile-train --data data/commect/measured.csv --tag qgb-development
```

The preparation command downloads and converts real public measurements; it does
not require MATLAB. It creates `data/commect/measured.csv`. You can also supply your
own measured trace using the schema below. The training command prints its run
folder. It trains only on training rows, selects one parameter candidate per
feature set using validation mean pinball loss across quantiles, and reports ONLY
validation performance. It does not refit on validation or score the test set.
`early_stopping=False` prevents a hidden randomly selected validation subset.

Then, when you are ready to inspect held-out results:

```powershell
uv run --no-sync python -m wireless_rtc quantile-evaluate --model runs/YOUR-TRAINING-RUN/model.pkl --data data/commect/measured.csv --tag qgb-heldout
```

For entirely new measured server traces, evaluate every row after warmup:

```powershell
uv run --no-sync python -m wireless_rtc quantile-evaluate --model runs/YOUR-TRAINING-RUN/model.pkl --data data/new-server-session.csv --partition all --tag qgb-server-test
```

Model files are Python pickle artifacts. Load only files you trained or trust.
The evaluator checks schema and the exact scikit-learn version. Dependency versions
are saved in the training manifest; use the same versions to reproduce a run.
Thread pools are limited to one during fitting and prediction.

## Input contract

UTF-8 CSV, one row per ordered measurement, with:

| Column | Meaning |
|---|---|
| `session_id` | Distinct run identifier; defaults to file stem if omitted |
| `timestamp_ms` | Numeric measurement availability time in milliseconds |
| `rtt_ms` | Measured RTT in milliseconds; the prediction label |
| `snr_db` | SNR available at this record; blank means missing |
| `snr_timestamp_ms` | Optional time of the underlying SNR measurement, in the same clock as `timestamp_ms` |
| `status` | `ok`, or an explicit failure such as `timeout` |
| `split` | Optional `train`, `validation` or `test`; supply for every row or none |

Small format example, not a training dataset:

```csv
session_id,timestamp_ms,rtt_ms,snr_db,status
walk-a,0,12.3,28.0,ok
walk-a,100,13.1,27.5,ok
walk-a,200,,26.0,timeout
```

Timestamps must strictly increase within each session. Multiple input files can be
passed after `--data`; distinguish sessions explicitly when names could collide.
Separate sessions are never joined into one history. Negative/nonfinite RTTs on
successful rows are rejected. Failures have no regression label, are counted in
the audit, and reset history. There is no zero filling or interpolation of RTTs.
The resulting regression metrics describe successful, predictable samples; they
do not establish reliability across censored failures.

If `snr_timestamp_ms` is omitted, the measurement is treated as available at its
row timestamp. Do not assign a fresh timestamp to an old cached SNR reading.
SNR timestamps later than their containing record are rejected. Use a past-only
join when aligning independently collected SNR and RTT logs.

Project probe CSVs using `wall_time_ns`, `actual_ms` and `wifi_snr_db` are also
accepted. These probes are serial; their row wall timestamp is a request-start
timestamp, so the prediction horizon in this format is start-to-start. The prior
RTT is already completed before the next serial probe. This alias is not suitable
for overlapping asynchronous requests without explicit availability timestamps.
Current Windows probes have no SNR and therefore require the RTT-only configuration
described below. The code never substitutes RSSI for SNR.

## Public data: COMMECT

The [COMMECT dataset](https://zenodo.org/records/16919305) documents cellular RTT and
SNR during mobile and stationary measurements. The actual ZIP contains
`latency_data.mat`, `downlink_data.mat`, `uplink_data.mat`, and the associated paper.
The latency MAT file stores MATLAB tables, not plain numeric arrays readable with
an ordinary SciPy `loadmat` call. The archive checksum is
`bf80eee0f22dd5bcee2ad092031cffd1` (MD5).

Prepare the data entirely in Python:

```powershell
uv run --no-sync python scripts/prepare_commect.py
```

The command downloads and verifies the archive if necessary, then decodes the
MATLAB tables using `mat-io==0.6.3`. Conversion dependencies are installed in a
separate `.deps/commect-converter` environment because this reader requires a newer
NumPy than the policy environment. The project environment is not upgraded.

Outputs are `data/commect/measured.csv` and `measured.source.json`, containing
46,311 mobility records and 35,824 stationary records (82,135 total). Source columns
are `Timestamp`, `Round_Trip_Time` (milliseconds), and `SNR` (dB). Each source table
becomes a separate session. Timestamps are converted to milliseconds relative to
that session's first measurement; row order and all observations are retained.
The reader's timezone/display-format warnings do not affect relative intervals
within a session. Satellite tables are excluded because they do not provide SNR.

The source manifest records the MAT-file hash, CSV hash, reader version and row
counts. Preparation refuses to replace a CSV containing different data. Rerunning
it with identical input is safe. To prepare only the mobile session:

```powershell
uv run --no-sync python scripts/prepare_commect.py --tables cellular_latency_mobility --output data/commect/mobility.csv
```

The earlier manual MATLAB exporter remains available as an alternative, but is
not required by this workflow.

Cellular data are method-development data, not proof of performance on campus
Wi-Fi. Keep source SNR/SINR definitions explicit. The dataset's measurement cadence
determines its prediction horizon; it cannot establish per-request accuracy at a
faster control cadence. Satellite traces do not supply the cellular SNR feature.

## Features and partitions

`configs/quantile_boosting.json` controls the experiment. It defaults to RTT history
only (no SNR), eight completed observations, quantiles 0.50/0.80/0.95, and two
parameter candidates. Earlier saved experiments may still contain the optional
RTT-plus-SNR comparison.

- `rtt`: RTT lags, median, mean, standard deviation, p95, maximum, endpoint trend,
  last observation interval, and history duration.
- `rtt_snr`: the same features plus SNR lags, observed mean/variation/trend,
  missingness fraction, and age of the latest available SNR observation.

All features come from the earlier completed history. The target row's RTT AND
SNR are excluded. This is intentionally a conservative next-sample forecast;
adding fresh pre-submission telemetry later requires the same availability
contract in both training and deployment. Missing SNR remains NaN and is handled
by the estimator. Both training and validation must contain observed historical
SNR for an `rtt_snr` experiment. If SNR is unavailable, copy the config, set
`feature_sets` to `["rtt"]`, and pass it with `--config`; do not invent SNR values.

With no explicit `split` column, chronological splitting uses the first 60% of
each session for training, next 20% for validation, and final 20% for testing.
Records are split BEFORE windows are built; each partition starts with a fresh
history, so windows never share observations across partitions. Long gaps
(default 10000 ms), failures, and session boundaries also reset history.

For multiple independent collection sessions, prefer explicit whole-session
assignments. Alternatively, `split_mode="session"` assigns entire sessions in
their input encounter order according to the fractions; enough sessions are
required for all three partitions. Do not randomly shuffle overlapping windows.

Frozen evaluation may use completed earlier measurements from the held-out stream
as history, but never updates weights or chooses parameters. The evaluator rejects
observations with the same session/time/value identity as training or validation.
Renaming or modifying source records defeats identity matching; the user must
still preserve genuinely independent sessions and an untouched final test.

## Outputs and interpretation

Training outputs `model.pkl`, `manifest.json`, `metrics.csv`, `metrics.json` and
`predictions.csv`. Evaluation produces the same reports without fitting a model.
The manifest records configuration, exact input hashes, dependency versions,
selected candidates, feature names and failure/warmup/gap audits. Reports include:

- `qgb_rtt` and `qgb_rtt_snr` on exactly the same target rows.
- Last-value, rolling-median, recent-max and matched rolling-quantile baselines
  using the same completed history and reset policy.
- MAE, signed bias, pinball loss, empirical quantile coverage/exceedance,
  step under/overprediction rates, mean excess steps and mean prefix length.

Independent quantile models can cross. Predictions are clipped to nonnegative
values and sorted across increasing quantiles, identically during validation
selection and later inference. This guarantees ordered outputs, not calibration.
CSV metrics render to two decimals; raw predictions and JSON retain precision.

`forecast_horizon_ms` is logged for every prediction. `control_ms` affects step
metrics only; it does not change the dataset sampling interval or generate a
fixed-time forecast. Compare p95 predictors using coverage and excess budgets,
not only MAE. Always report the sample counts excluded by failures and warmup.

The `QuantileBundle.predict(history, feature_set)` helper can support later live
integration, but this change does not replace RTC's live predictor or claim an
improvement in solve rate. Training the Mostafavi-style MDN is the next separate
implementation; it can reuse the data and feature modules.

Reference: [scikit-learn quantile histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
