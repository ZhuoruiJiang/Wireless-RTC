# Repository guidance

Read `docs/server_handoff.md` before continuing the experiment or Mininet-WiFi work. Read `docs/artifact_tracking.md` before saving or publishing results.

- The `mininet-wifi` branch is the development branch for reproducible wireless emulation. Preserve the real-network experiments and raw data as reference. Mininet-WiFi is not implemented yet at this checkpoint.
- Track experiment outputs, datasets, reports, and checkpoints in Git; use Git LFS for large traces and binary artifacts. Do not commit credentials, environments, or dependency caches. Generated files are not uploaded automatically: stage, commit, and push them explicitly.
- Never move, delete, or rewrite raw runs to tidy folders or relabel conditions without the user's instruction. Reporting labels belong in selection configuration files.
- Summary procedures must use explicit run-selection configuration files. Do not silently select all folders, the latest report, or incompatible task sets. See `docs/result_selection.md`.
- Current rollout tasks are `catcher_v3`, `hard_lunar_lander`, and `mjc_walker`. Preserve task compatibility checks.
- Main prediction tables include last response, recent maximum, and rolling p95. Rolling p50/p70/p80/p90 and the frozen QGB checkpoint are internal comparisons. Do not retrain QGB. The Fugu-inspired MLP is deferred.
- Output metrics named `_pct` use 0--100. Preserve raw numerical precision; round only for presentation.
- Keep server inference time distinct from full policy-response time. Vanilla RTC predicts inference time; wireless RTC predicts full response time. Actual arrival determines usable action indices. Buffer deadline misses are distinct from prediction underestimation.
- Retain the logical TCP client/server boundary in emulation, even on one machine. Ensure traffic traverses the emulated wireless path, rather than localhost. Label simulated results separately from real Wi-Fi measurements.
- Run relevant tests after code changes. Current suite: `uv run --no-sync python -m unittest discover -s tests -q`.
