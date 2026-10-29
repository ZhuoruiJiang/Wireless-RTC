"""Causal feature construction: only completed earlier records enter a forecast."""
from collections import deque
import math
import statistics

from wireless_rtc.prediction.rolling import quantile


def feature_names(history, feature_set):
    if feature_set not in {"rtt", "rtt_snr"}:
        raise ValueError("feature_set must be rtt or rtt_snr")
    names = [f"rtt_lag_{i}_ms" for i in range(1, history+1)]
    names += ["rtt_median_ms", "rtt_mean_ms", "rtt_std_ms", "rtt_p95_ms", "rtt_max_ms", "rtt_trend_ms",
              "last_sample_interval_ms", "history_duration_ms"]
    if feature_set == "rtt_snr":
        names += [f"snr_lag_{i}_db" for i in range(1, history+1)]
        names += ["snr_mean_db", "snr_std_db", "snr_trend_db", "snr_missing_fraction", "snr_age_ms"]
    return names


def feature_values(history, feature_set):
    """Public inference helper; history must contain completed successful samples."""
    if len(history) < 2 or any(r.rtt_ms is None or r.status != "ok" for r in history):
        raise ValueError("Features need at least two completed successful RTT samples")
    rtts = [r.rtt_ms for r in history]
    values = list(reversed(rtts)) + [statistics.median(rtts), statistics.mean(rtts), statistics.pstdev(rtts),
              quantile(rtts, .95), max(rtts), rtts[-1]-rtts[0],
              history[-1].timestamp_ms-history[-2].timestamp_ms,
              history[-1].timestamp_ms-history[0].timestamp_ms]
    if feature_set == "rtt_snr":
        observed = [(r.snr_timestamp_ms if r.snr_timestamp_ms is not None else r.timestamp_ms, r.snr_db)
                    for r in history if r.snr_db is not None]
        snrs = [v for _,v in observed]
        values += [r.snr_db if r.snr_db is not None else math.nan for r in reversed(history)]
        values += [statistics.mean(snrs) if snrs else math.nan, statistics.pstdev(snrs) if snrs else math.nan,
                   snrs[-1]-snrs[0] if len(snrs)>1 else math.nan, 1-len(snrs)/len(history),
                   history[-1].timestamp_ms-observed[-1][0] if observed else math.nan]
    elif feature_set != "rtt":
        raise ValueError("Unknown feature set")
    return values


def build_examples(sessions, history, max_gap_ms, partition=None):
    """Failures, long gaps and split boundaries reset history; no cross-session windows."""
    examples, audit = [], dict(records=0, failed_records=0, warmup_records=0, gap_resets=0)
    for rows in sessions.values():
        context = deque(maxlen=history)
        previous = None
        for record in rows:
            if partition is not None and record.split != partition:
                context.clear()
                previous = None
                continue
            audit["records"] += 1
            if previous is not None:
                if record.split != previous.split:
                    context.clear()
                if record.timestamp_ms-previous.timestamp_ms > max_gap_ms:
                    context.clear()
                    audit["gap_resets"] += 1
            previous = record
            if record.status != "ok" or record.rtt_ms is None:
                audit["failed_records"] += 1
                context.clear()
                continue
            if len(context) == history:
                examples.append(dict(target=record, history=list(context)))
            else:
                audit["warmup_records"] += 1
            context.append(record)
    audit["examples"] = len(examples)
    return examples, audit
