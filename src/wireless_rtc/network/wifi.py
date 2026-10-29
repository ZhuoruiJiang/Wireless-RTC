"""Cached Windows link telemetry. No scans, inferred RSSI, or invented SNR."""
import csv
import math
import platform
import re
import subprocess
import threading
import time


WIFI_FIELDS = ["wifi_status", "wifi_error", "wifi_sample_start_ns", "wifi_sample_end_ns",
               "wifi_available_monotonic_ns", "wifi_interface", "wifi_state", "wifi_ssid",
               "wifi_bssid", "wifi_band", "wifi_channel", "wifi_radio_type", "wifi_signal_pct",
               "wifi_rssi_dbm", "wifi_rx_mbps", "wifi_tx_mbps", "wifi_noise_dbm", "wifi_snr_db"]


def parse_interfaces(text, interface=None):
    """Parse English netsh output; report unsupported/localized output explicitly."""
    adapters, current = [], None
    for line in text.splitlines():
        match = re.match(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        key, value = match.groups()
        if key == "Name":
            current = {}
            adapters.append(current)
        if current is not None:
            current[key] = value
    selected = [a for a in adapters if interface is None or a.get("Name") == interface]
    if len(selected) != 1:
        raise ValueError("Select one adapter with --wifi-interface; netsh must return English interface data")
    adapter = selected[0]
    result = {"wifi_interface": adapter["Name"], "wifi_state": adapter.get("State", "unknown")}
    if result["wifi_state"] != "connected":
        result["wifi_status"] = "disconnected"
        return result
    result["wifi_status"] = "ok"
    for source, target in [("SSID", "ssid"), ("AP BSSID", "bssid"), ("Band", "band"),
                           ("Radio type", "radio_type")]:
        if source in adapter:
            result[f"wifi_{target}"] = adapter[source]
    if "wifi_bssid" not in result and "BSSID" in adapter:
        result["wifi_bssid"] = adapter["BSSID"]
    for source, target in [("Channel", "channel"), ("Signal", "signal_pct"), ("Rssi", "rssi_dbm"),
                           ("Receive rate (Mbps)", "rx_mbps"), ("Transmit rate (Mbps)", "tx_mbps")]:
        if source in adapter:
            value = float(adapter[source].rstrip("% "))
            if not math.isfinite(value):
                raise ValueError(f"Invalid Wi-Fi metric: {source}")
            result[f"wifi_{target}"] = value
    if "wifi_signal_pct" not in result and "wifi_rssi_dbm" not in result:
        raise ValueError("Connected adapter returned no signal metric; check netsh permissions/output language")
    # netsh does not expose a noise floor: leave noise and SNR missing.
    return result


class WifiSampler:
    """Persist each poll and expose only a sample completed before snapshot()."""

    def __init__(self, path, interface=None, interval_ms=1000):
        if platform.system() != "Windows":
            raise ValueError("--wifi currently supports Windows netsh; use iw telemetry on Linux")
        if not math.isfinite(interval_ms) or interval_ms < 100:
            raise ValueError("Wi-Fi polling interval must be finite and at least 100 ms")
        self.path, self.interface, self.interval = path, interface, interval_ms/1000
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = {"wifi_status": "pending"}

    def _poll(self):
        row = dict(wifi_sample_start_ns=time.time_ns())
        try:
            result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True,
                                    text=True, errors="replace", timeout=5,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                raise ValueError(result.stdout.strip() or result.stderr.strip() or f"netsh exit {result.returncode}")
            row.update(parse_interfaces(result.stdout, self.interface))
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            row.update(wifi_status="error", wifi_error=str(exc))
        row.update(wifi_sample_end_ns=time.time_ns(), wifi_available_monotonic_ns=time.monotonic_ns())
        self.writer.writerow(row)
        self.file.flush()
        with self.lock:
            self.latest = row

    def _run(self):
        while not self.stop.wait(self.interval):
            self._poll()

    def snapshot(self):
        with self.lock:
            row = dict(self.latest)
        available = row.get("wifi_available_monotonic_ns")
        row["wifi_age_ms"] = (time.monotonic_ns()-available)/1e6 if available is not None else None
        return row

    def __enter__(self):
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=WIFI_FIELDS)
        self.writer.writeheader()
        self._poll()  # Outside all measured RPCs; do not start with an unknown sample.
        self.thread = threading.Thread(target=self._run, name="wifi-telemetry", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.file.close()
