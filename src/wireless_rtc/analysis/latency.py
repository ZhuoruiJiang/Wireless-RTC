"""Request-weighted live latency summaries; missing measurements are not zero."""
import math


SUMMARY_METRICS = {
    "rtt": ("actual_ms", "average_rtt_ms"),
    "injected_delay": ("injected_delay_ms", "average_injected_delay_ms"),
    "delay_sleep": ("delay_sleep_ms", "average_delay_sleep_ms"),
    "prediction": ("prediction_absolute_error_ms", "prediction_mae_ms"),
    "rtt_prediction": ("rtt_prediction_absolute_error_ms", "rtt_prediction_mae_ms"),
}


def valid_measurement(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


class LatencyStats:
    def __init__(self):
        self.sums = {name: 0. for name in SUMMARY_METRICS}
        self.counts = {name: 0 for name in SUMMARY_METRICS}

    def observe(self, row):
        if row.get("status") != "ok":
            return
        values = dict(row, prediction_absolute_error_ms=None, rtt_prediction_absolute_error_ms=None)
        target_field = {"inference_ms":"inference_ms", "response_rtt_ms":"actual_ms"}.get(row.get("predictor_basis"))
        predicted = row.get("predicted_ms")
        if target_field is not None and valid_measurement(predicted):
            target = row.get(target_field)
            if valid_measurement(target):
                values["prediction_absolute_error_ms"] = abs(predicted-target)
            if valid_measurement(row.get("actual_ms")):
                values["rtt_prediction_absolute_error_ms"] = abs(predicted-row["actual_ms"])
        for name, (field, _) in SUMMARY_METRICS.items():
            value = values.get(field)
            if valid_measurement(value):
                self.sums[name] += value
                self.counts[name] += 1

    def summary(self):
        result = {}
        for name, (_, average) in SUMMARY_METRICS.items():
            count = self.counts[name]
            result.update({average: self.sums[name]/count if count else None,
                           f"{name}_samples": count})
        return result


def combine_latency(rows):
    """Pool episode/task means by their measured request counts."""
    result = {}
    for name, (_, average) in SUMMARY_METRICS.items():
        samples = [(row[average], row.get(f"{name}_samples", 0)) for row in rows
                   if row.get(average) is not None and row.get(f"{name}_samples", 0) > 0]
        count = sum(n for _, n in samples)
        result.update({average: math.fsum(value*n for value, n in samples)/count if count else None,
                       f"{name}_samples": count})
    return result
