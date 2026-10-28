"""Request-weighted live latency summaries; missing measurements are not zero."""
import math


LATENCY_METRICS = {
    "rtt": ("actual_ms", "average_rtt_ms"),
    "injected_delay": ("injected_delay_ms", "average_injected_delay_ms"),
    "delay_sleep": ("delay_sleep_ms", "average_delay_sleep_ms"),
}


class LatencyStats:
    def __init__(self):
        self.sums = {name: 0. for name in LATENCY_METRICS}
        self.counts = {name: 0 for name in LATENCY_METRICS}

    def observe(self, row):
        if row.get("status") != "ok":
            return
        for name, (field, _) in LATENCY_METRICS.items():
            value = row.get(field)
            if (not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value) and value >= 0):
                self.sums[name] += value
                self.counts[name] += 1

    def summary(self):
        result = {}
        for name, (_, average) in LATENCY_METRICS.items():
            count = self.counts[name]
            result.update({average: self.sums[name]/count if count else None,
                           f"{name}_samples": count})
        return result


def combine_latency(rows):
    """Pool episode/task means by their measured request counts."""
    result = {}
    for name, (_, average) in LATENCY_METRICS.items():
        samples = [(row[average], row.get(f"{name}_samples", 0)) for row in rows
                   if row.get(average) is not None and row.get(f"{name}_samples", 0) > 0]
        count = sum(n for _, n in samples)
        result.update({average: math.fsum(value*n for value, n in samples)/count if count else None,
                       f"{name}_samples": count})
    return result
