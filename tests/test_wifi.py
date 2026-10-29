import csv
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from wireless_rtc.network.wifi import parse_interfaces
from wireless_rtc.network.probe import run_probe
from wireless_rtc.server.service import make_server


CONNECTED = """
    Name : Wi-Fi
    State : connected
    SSID : campus
    AP BSSID : 08:7b:87:aa:bb:cc
    Band : 6 GHz
    Channel : 181
    Radio type : 802.11ax
    Receive rate (Mbps) : 1201
    Transmit rate (Mbps) : 1201
    Signal : 80%
    Rssi : -62
"""


class WifiTests(unittest.TestCase):
    def test_actual_rssi_and_missing_snr(self):
        row = parse_interfaces(CONNECTED)
        self.assertEqual(row["wifi_rssi_dbm"], -62)
        self.assertEqual(row["wifi_signal_pct"], 80)
        self.assertEqual(row["wifi_bssid"], "08:7b:87:aa:bb:cc")
        self.assertNotIn("wifi_snr_db", row)
        self.assertNotIn("wifi_noise_dbm", row)
        self.assertNotIn("wifi_rssi_dbm", parse_interfaces(CONNECTED.replace("    Rssi : -62", "")))

    def test_disconnect_and_ambiguous_or_unsupported_output(self):
        row = parse_interfaces("Name : Wi-Fi\nState : disconnected\n")
        self.assertEqual(row["wifi_status"], "disconnected")
        self.assertNotIn("wifi_rssi_dbm", row)
        for text in ["location access denied", CONNECTED + CONNECTED.replace("Wi-Fi", "Wi-Fi 2")]:
            with self.assertRaises(ValueError):
                parse_interfaces(text)
        self.assertEqual(parse_interfaces(CONNECTED + CONNECTED.replace("Wi-Fi", "Wi-Fi 2"),
                                         "Wi-Fi 2")["wifi_interface"], "Wi-Fi 2")

    def test_probe_wifi_samples_are_causal_and_unavailable_data_is_explicit(self):
        server = make_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for output in [CONNECTED, "Access denied"]:
                with self.subTest(output=output), tempfile.TemporaryDirectory() as temp, (
                        patch("wireless_rtc.network.wifi.platform.system", return_value="Windows")), (
                        patch("wireless_rtc.network.wifi.subprocess.run",
                              return_value=SimpleNamespace(returncode=0, stdout=output, stderr=""))):
                    path = run_probe("127.0.0.1", server.server_address[1], Path(temp)/"probe",
                                     samples=2, interval_ms=0, wifi=True)
                    with Path(path).open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        self.assertEqual(row["status"], "ok")
                        self.assertEqual(float(row["hold_ms"]), 0)
                        self.assertLessEqual(int(row["wifi_available_monotonic_ns"]),
                                             int(row["request_start_monotonic_ns"]))
                        self.assertEqual(row["wifi_snr_db"], "")
                        self.assertEqual(row["wifi_status"], "ok" if output == CONNECTED else "error")
                    self.assertTrue((Path(temp)/"probe"/"wifi.csv").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
