"""Ordered, measured RTT records shared by learned prediction experiments."""
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path


PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class Record:
    session: str
    timestamp_ms: float
    rtt_ms: float | None
    snr_db: float | None = None
    split: str = ""
    status: str = "ok"
    snr_timestamp_ms: float | None = None

    @property
    def identity(self):
        # Split-independent identity detects reused observations in held-out tests.
        return hashlib.sha256(json.dumps(
            [self.session, self.timestamp_ms, self.rtt_ms, self.snr_db, self.status, self.snr_timestamp_ms],
            separators=(",", ":")).encode()).hexdigest()


def finite(value, field, optional=False):
    if value is None or str(value).strip().lower() in {"", "nan", "na", "?", "null"}:
        if optional:
            return None
        raise ValueError(f"Missing {field}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {field}")
    return value


def read_records(paths):
    """Accept canonical CSV or this project's probe CSV. Never shuffle or fill RTTs."""
    sessions, provenance = {}, []
    for path in map(Path, paths):
        if not path.is_file():
            raise ValueError(f"Measured CSV not found: {path}. data/measured.csv is an example path; "
                             "prepare a real dataset first (see docs/quantile_boosting.md).")
        provenance.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            if not ({"timestamp_ms", "wall_time_ns"} & columns) or not ({"rtt_ms", "actual_ms"} & columns):
                raise ValueError(f"{path}: require timestamp_ms and rtt_ms (or probe wall_time_ns and actual_ms)")
            for line, row in enumerate(reader, 2):
                try:
                    session = row.get("session_id") or path.stem
                    timestamp = (finite(row["timestamp_ms"], "timestamp_ms") if "timestamp_ms" in columns
                                 else finite(row["wall_time_ns"], "wall_time_ns")/1e6)
                    status = row.get("status") or "ok"
                    rtt = finite(row.get("rtt_ms", row.get("actual_ms")), "rtt_ms", optional=status != "ok")
                    if rtt is not None and rtt < 0:
                        raise ValueError("Negative RTT")
                    snr = finite(row.get("snr_db", row.get("wifi_snr_db")), "snr_db", optional=True)
                    snr_time = finite(row.get("snr_timestamp_ms"), "snr_timestamp_ms", optional=True)
                    if snr_time is not None and snr_time > timestamp:
                        raise ValueError("SNR timestamp is in the future relative to its record")
                    if snr is not None and snr_time is None:
                        snr_time = timestamp
                    partition = row.get("split", "").strip()
                    if partition and partition not in PARTITIONS:
                        raise ValueError("split must be train, validation or test")
                    record = Record(session, timestamp, rtt if status == "ok" else None, snr, partition, status, snr_time)
                    previous = sessions.setdefault(session, [])
                    if previous and timestamp <= previous[-1].timestamp_ms:
                        raise ValueError("Session timestamps must strictly increase; use distinct session_id values for separate runs")
                    previous.append(record)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line}: {exc}") from exc
    if not sessions:
        raise ValueError("Empty dataset")
    return sessions, provenance


def partition_records(sessions, config):
    """Split raw records BEFORE making windows, so partitions share no observations."""
    records = [r for rows in sessions.values() for r in rows]
    explicit = [bool(r.split) for r in records]
    if any(explicit):
        if not all(explicit):
            raise ValueError("Either every row has a split, or no row does")
        for rows in sessions.values():
            order = [PARTITIONS.index(r.split) for r in rows]
            if order != sorted(order):
                raise ValueError("Within-session explicit splits must be chronological")
        return sessions
    mode = config["split_mode"]
    train, validation = config["train_fraction"], config["validation_fraction"]
    if mode == "session":
        keys = list(sessions)
        a, b = int(len(keys)*train), int(len(keys)*(train+validation))
        if not 0 < a < b < len(keys):
            raise ValueError("Session split needs enough ordered sessions for all three partitions")
        return {key: [replace(r, split=PARTITIONS[0 if i<a else 1 if i<b else 2]) for r in sessions[key]]
                for i, key in enumerate(keys)}
    if mode != "chronological":
        raise ValueError("split_mode must be chronological or session")
    result = {}
    for key, rows in sessions.items():
        a, b = int(len(rows)*train), int(len(rows)*(train+validation))
        result[key] = [replace(r, split=PARTITIONS[0 if i<a else 1 if i<b else 2]) for i,r in enumerate(rows)]
    return result


def export_records(path, sessions):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["session_id", "timestamp_ms", "rtt_ms", "snr_db", "status", "split", "snr_timestamp_ms"])
        writer.writeheader()
        for rows in sessions.values():
            for r in rows:
                writer.writerow(dict(session_id=r.session, timestamp_ms=r.timestamp_ms, rtt_ms=r.rtt_ms,
                                     snr_db=r.snr_db, status=r.status, split=r.split, snr_timestamp_ms=r.snr_timestamp_ms))
