# Choose the exact runs for each report procedure

All three report commands now use explicit configuration files. **Only listed
directories are included.** They do not discover more runs from a folder, consult
`session_map.json`, or choose a latest report automatically. Raw traces, original
manifests and folders are never changed by report selection.

## Table 1: network characteristics

Edit `configs/network_summary.json`, then run:

```powershell
uv run --no-sync python -m wireless_rtc network-summary --config configs/network_summary.json
```

The current file lists the two dorm sessions and hallway session `064156`. To use
one repetition, remove the other path from this configuration. To pool repetitions,
list each desired run once. To replace an environment, replace its list of paths.
Old runs can remain wherever you store them; unlisted runs do not enter the table.

The schema is:

```json
{
  "schema_version": 1,
  "base_dir": "..",
  "split": "test",
  "workload": "policy",
  "runs": {
    "strong": ["runs/path/to/first-selected-run", "runs/path/to/second-selected-run"],
    "weak": ["runs/path/to/selected-hallway-run"]
  }
}
```

Each path names the **run directory containing `manifest.json`**, not `trace.csv`.
`base_dir` is relative to the configuration file. For files in `configs/`, `".."`
means the repository root. Run paths are relative to that base directory; absolute
paths also work. No wildcard expansion is performed.

The group names in `runs` set the report labels. Explicit selection is authoritative
even if an archived manifest has an older setting or split label. The output audit
preserves those original labels. Choosing a development run does not make its data
held-out; the selection file records your cohort, not new measurement provenance.

## Table 2: offline prediction

Edit `configs/midterm_prediction.json`, then run:

```powershell
uv run --no-sync python -m wireless_rtc prediction-summary --config configs/midterm_prediction.json
```

The default includes `"runs_config": "network_summary.json"`, so Tables 1 and 2
use exactly the same configured sessions. This filename is relative to the
prediction config file, not `base_dir`. The command does not need Table 1 to have
been generated first. History, control interval and the existing frozen model
checkpoint remain configured in the prediction file.

To evaluate a different subset, copy the network selection to another file and
change `runs_config` to that filename. Alternatively, remove `runs_config` and put
the `runs` object directly into the prediction config, with the appropriate
`base_dir`. Do not specify both. Selection references are one level deep to avoid
cycles. No model is retrained.

`--table1 <exact-summary.json>` remains available as an intentional override for
reproducing an earlier Table 1 cohort. It never selects an older report implicitly.

## Tables 3 and 4: rollout summaries

Edit `configs/rollout_summary.json`, then run:

```powershell
uv run --no-sync python -m wireless_rtc rollout-summary --config configs/rollout_summary.json
```

Its `runs` object lists the precise block directories to include under each setting.
Add a new block's path after collecting it. For repeated attempts of the same
block, choose the completed attempt you want to use. Duplicate block numbers in a
setting are rejected because they reuse the same episode seeds. For a separate
repeat analysis, copy the config and choose the other attempt.

The initial config lists the new strong block 1 and the four completed weak blocks.
Append strong blocks 2--4 when available. Selected blocks must be completed and
must use matching task sets, checkpoint hashes, predictors, timing and seed
settings. There is no automatic intersection of tasks or task averaging across
different benchmarks. No earlier strong development sweeps are selected.

To inspect a finished setting alone:

```powershell
uv run --no-sync python -m wireless_rtc rollout-summary --config configs/rollout_summary.json --settings weak
```

To summarize a deliberately selected subset of completed blocks, add
`--allow-partial` (or set `"allow_partial": true` in the config). Missing block
numbers are recorded. This does **not** skip a running/failed block explicitly
listed in the file; remove it from the subset config or wait for it to complete.

## Validation and reproducibility

- Missing paths, duplicate paths (including the same path in two groups), and
  unfinished selected runs produce errors. No replacement run is chosen silently.
- Workload/checkpoint compatibility checks remain active.
- Each successful output includes `selection.json` with the configuration snapshot,
  resolved paths, original labels and file hashes. Table 1's summary and Table 2's
  manifest also record selection information; rollout summaries name the tasks.
- Output folders are timestamped and preserved. `--experiment` and `--output-root`
  organize outputs; configured paths determine inputs.
- Old reports keep their original cohorts. Regenerate a report after changing its
  selection config. Config edits do not rewrite an old report or the open LaTeX file.

The collection commands `network-characterize` and `rtc-live` still create new
sessions. Their measurement settings are independent of these analysis-selection
files.
