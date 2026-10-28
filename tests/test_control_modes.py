"""Compare causal predictor inputs under identical server-injected waiting."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wireless_rtc.client.live import run_live_episode, run_remote
from wireless_rtc.control_modes import CONTROL_MODES, predictors_for
from wireless_rtc.network.transport import RPCClient
from wireless_rtc.prediction.rolling import RollingPredictor
from wireless_rtc.server.delay import NormalResponseDelay
from wireless_rtc.server.service import make_server


class ModePolicy:
    def infer(self, request):
        return dict(observation_tick=request["observation_tick"],inference_ms=2.,actions=[[0.]]*64)

    def describe(self):
        return {"protocol_version":1}


class ModeTests(unittest.TestCase):
    def test_guidance_learns_inference_or_total_delay_without_future_leakage(self):
        for method in CONTROL_MODES:
            with self.subTest(method=method):
                server = make_server("127.0.0.1",0,ModePolicy(),NormalResponseDelay(mean_ms=35,std_ms=0,max_ms=100))
                thread = threading.Thread(target=server.serve_forever,daemon=True)
                thread.start()
                predictor = RollingPredictor(kind="last",window=1,initial_ms=3)
                requests = []
                def step(state,action):
                    return [state+1],state+1,False,False
                try:
                    with RPCClient("127.0.0.1",server.server_address[1]) as client:
                        result = run_live_episode(client,[0],0,step,[[0.]]*64,predictor,5,1,24,0,
                                                  method,requests.append,lambda row:None,
                                                  initial_inference_ms=2 if method == "rtc" else None)
                finally:
                    server.shutdown(); server.server_close(); thread.join(timeout=3)
                self.assertEqual(result["status"],"ok")
                self.assertGreaterEqual(len(requests),2)
                self.assertTrue(all(r["injected_delay_ms"] == 35 for r in requests))
                self.assertTrue(all(r["predictor_basis"] == CONTROL_MODES[method] for r in requests))
                if method == "rtc":
                    self.assertTrue(all(r["predicted_ms"] == 2 for r in requests))
                    self.assertTrue(all(r["predictor_observation_ms"] == 2 for r in requests))
                    self.assertTrue(all(r["actual_ms"] >= 34 for r in requests))
                elif method == "wireless_rtc":
                    self.assertEqual(requests[0]["predicted_ms"],3)
                    self.assertEqual(requests[1]["predicted_ms"],requests[0]["actual_ms"])
                    self.assertTrue(all(r["predictor_observation_ms"] == r["actual_ms"] for r in requests))
                else:
                    self.assertTrue(all(r["predicted_ms"] is None and r["predicted_steps"] == 0 for r in requests))
                    self.assertFalse(predictor.history)

    def test_naive_has_one_condition_independent_of_predictor_count(self):
        config = {"predictors":{"a":{},"b":{}}}
        self.assertEqual(list(predictors_for("naive",config)),[("none",None)])
        self.assertEqual(len(list(predictors_for("rtc",config))),2)
        self.assertEqual(len(list(predictors_for("wireless_rtc",config))),2)

    def test_old_server_protocol_is_rejected_before_mislabeled_experiment(self):
        sim = SimpleNamespace(jax=SimpleNamespace(devices=lambda:["cpu"]),jnp=None,np=None)
        config = json.loads(Path("configs/rtc_remote.json").read_text())
        server = make_server("127.0.0.1",0,ModePolicy())
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp,patch(
                    "wireless_rtc.simulation.kinetix.KinetixSimulator",return_value=sim):
                output = Path(temp)/"run"
                with self.assertRaisesRegex(ValueError,"update and restart"):
                    run_remote(config,"127.0.0.1",server.server_address[1],output,{})
                manifest = json.loads((output/"manifest.json").read_text())
                self.assertEqual(manifest["status"],"error")
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)
