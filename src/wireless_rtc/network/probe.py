import csv
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import time
from wireless_rtc.network.transport import RPCClient
from wireless_rtc.network.delays import generate_delays
from wireless_rtc.prediction.rolling import make_predictor
from wireless_rtc.network.wifi import WIFI_FIELDS, WifiSampler


def run_probe(host, port, output, samples=100, interval_ms=100, payload_bytes=1024,
              delay_config=None, predictor_config=None, timeout=10, seed=42,
              wifi=False, wifi_interface=None, wifi_interval_ms=1000):
    if samples < 1 or interval_ms < 0 or not 0 <= payload_bytes <= 1000000:
        raise ValueError("Invalid probe settings")
    output = Path(output)
    sampler = WifiSampler(output/"wifi.csv", wifi_interface, wifi_interval_ms) if wifi else None
    output.mkdir(parents=True, exist_ok=False)
    predictor_config = predictor_config or {"kind": "max", "window": 32}
    predictor = make_predictor(predictor_config)
    holds = generate_delays(delay_config or {"kind":"fixed", "value_ms":0}, samples, seed)
    manifest = dict(kind="network_probe", host=host, port=port, samples=samples,
                    interval_ms=interval_ms, payload_bytes=payload_bytes, timeout=timeout,
                    delay_config=delay_config, predictor=predictor_config,
                    platform=platform.platform(), seed=seed,
                    wifi=dict(enabled=wifi,interface=wifi_interface,interval_ms=wifi_interval_ms,
                              source="netsh wlan show interfaces",noise_and_snr_available=False),
                    note="hold_ms is artificial SERVER processing delay, not measured WiFi delay")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fields = ["index","wall_time_ns","request_start_monotonic_ns","request_end_monotonic_ns",
              "status","actual_ms","predicted_ms","server_ms","hold_ms","error"]
    if wifi:
        fields += WIFI_FIELDS + ["wifi_age_ms"]
    client = None
    with (sampler if sampler is not None else nullcontext()) as telemetry, (
            (output/"trace.csv").open("w", newline="", encoding="utf-8")) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        try:
            for i, hold in enumerate(holds):
                row = dict(index=i, wall_time_ns=time.time_ns(), predicted_ms=predictor.predict(), hold_ms=hold)
                if telemetry is not None:
                    row.update(telemetry.snapshot())
                row["request_start_monotonic_ns"] = time.monotonic_ns()
                try:
                    if client is None:
                        client = RPCClient(host, port, timeout)
                    response, elapsed = client.request("probe", payload="x"*payload_bytes, hold_ms=hold)
                    row.update(status="ok", actual_ms=elapsed, server_ms=response["server_ms"])
                    predictor.update(elapsed)
                except (OSError, EOFError, ValueError) as exc:
                    row.update(status="error", error=str(exc))
                    if client is not None:
                        client.close()
                        client = None  # never reuse a timed-out stream
                row["request_end_monotonic_ns"] = time.monotonic_ns()
                writer.writerow(row)
                f.flush()
                time.sleep(interval_ms/1000)
        finally:
            if client is not None:
                client.close()
    return str(output/"trace.csv")
