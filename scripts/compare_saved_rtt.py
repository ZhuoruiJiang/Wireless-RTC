"""Development-only causal predictor comparison on saved wireless request RTTs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics

from wireless_rtc.analysis.evaluate import evaluate
from wireless_rtc.analysis.formatting import ReportWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New comparison CSV path")
    parser.add_argument("--control-ms", type=float, default=20)
    args = parser.parse_args()
    episodes = []
    for path in sorted(args.run.glob("tasks/*/wireless_rtc-*/requests.csv")):
        grouped = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["status"] != "ok":
                    raise ValueError(f"{path}: censored/failed request requires separate analysis")
                grouped[row["episode"]].append(float(row["actual_ms"]))
        episodes.extend(grouped.values())
    if not episodes:
        raise ValueError("No wireless request episodes found")
    candidates = {
        "recent_max_w32": dict(kind="max", window=32),
        "p95_w32": dict(kind="quantile", window=32, quantile_level=.95),
        "last_value": dict(kind="last"),
        "median_w8": dict(kind="quantile", window=8, quantile_level=.5),
        "p80_w8": dict(kind="quantile", window=8, quantile_level=.8),
        "p90_w8": dict(kind="quantile", window=8, quantile_level=.9),
    }
    summaries = []
    for name, config in candidates.items():
        rows = [row for episode in episodes for row in
                evaluate(episode, config, control_ms=args.control_ms, warmup=0)[0]]
        summaries.append(dict(
            predictor=name, samples=len(rows), episodes=len(episodes),
            mae_ms=statistics.mean(abs(r["predicted_ms"]-r["actual_ms"]) for r in rows),
            excess_steps_pct=100*statistics.mean(r["predicted_steps"]>r["actual_steps"] for r in rows),
            insufficient_steps_pct=100*statistics.mean(r["predicted_steps"]<r["actual_steps"] for r in rows)))
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = ReportWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Development diagnostics only; no new solve-rate claims. Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
