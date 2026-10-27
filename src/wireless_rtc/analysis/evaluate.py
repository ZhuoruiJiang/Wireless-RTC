import csv
import json
import math
from pathlib import Path
import statistics
from wireless_rtc.client.schedule import delay_steps
from wireless_rtc.prediction.rolling import make_predictor, quantile


def evaluate(delays, predictor_config, control_ms=20, warmup=32, q=.95):
    if warmup < 0 or not 0 < q < 1:
        raise ValueError("Invalid warmup or scoring quantile")
    predictor, rows = make_predictor(predictor_config), []
    for index, actual in enumerate(delays):
        predicted = predictor.predict()
        predictor.update(actual)  # current target only becomes visible AFTER prediction
        error = actual - predicted
        rows.append(dict(index=index, actual_ms=actual, predicted_ms=predicted,
                         actual_steps=delay_steps(actual, control_ms),
                         predicted_steps=delay_steps(predicted, control_ms),
                         scored=index >= warmup,
                         pinball=max(q * error, (q - 1) * error)))
    scored = [r for r in rows if r["scored"]]
    if not scored:
        raise ValueError("No scored samples; reduce warmup or increase trace length")
    mean = statistics.mean
    metrics = dict(samples=len(scored), warmup_samples=min(warmup, len(rows)),
        mae_ms=mean(abs(r["actual_ms"]-r["predicted_ms"]) for r in scored),
        bias_ms=mean(r["predicted_ms"]-r["actual_ms"] for r in scored),
        exceedance_rate=mean(r["actual_ms"] > r["predicted_ms"] for r in scored),
        step_exceedance_rate=mean(r["actual_steps"] > r["predicted_steps"] for r in scored),
        mean_prefix_steps=mean(r["predicted_steps"] for r in scored),
        mean_overestimate_ms=mean(max(0,r["predicted_ms"]-r["actual_ms"]) for r in scored),
        pinball_loss_ms=mean(r["pinball"] for r in scored), scoring_quantile=q,
        actual_p50_ms=quantile([r["actual_ms"] for r in scored], .5),
        actual_p95_ms=quantile([r["actual_ms"] for r in scored], .95))
    return rows, metrics


def save_evaluation(directory, rows, metrics, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_plot(directory / "prediction.svg", rows)


def write_plot(path, rows):
    ymax = max(1, max(max(r["actual_ms"], r["predicted_ms"]) for r in rows)) * 1.1
    def points(field):
        return " ".join(f'{65+i/max(1,len(rows)-1)*900:.2f},{250-r[field]/ymax*210:.2f}'
                        for i,r in enumerate(rows))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="300" viewBox="0 0 1000 300">',
        '<rect width="1000" height="300" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#222">',
        '<text x="65" y="22">Response delay: actual (blue), predicted (orange)</text>',
        '<path d="M65 40 V250 H965" fill="none" stroke="#444"/>',
        f'<text x="5" y="45">{ymax:.0f} ms</text><text x="20" y="250">0 ms</text>',
        f'<text x="65" y="280">Request index: 0 to {len(rows)-1}; prediction precedes observation</text>',
        '</g>',
        f'<polyline points="{points("actual_ms")}" fill="none" stroke="#1976b8" stroke-width="1.3"/>',
        f'<polyline points="{points("predicted_ms")}" fill="none" stroke="#d66a17" stroke-width="1.3"/>',
        '</svg>']
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def read_trace(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty trace")
    if any(r.get("status", "ok") != "ok" for r in rows):
        raise ValueError("Timeouts are censored delays; report separately rather than silently drop")
    delays = [float(r["actual_ms"]) for r in rows]
    if any(not math.isfinite(x) or x < 0 for x in delays):
        raise ValueError("Trace has invalid delays")
    return delays
