"""Sample statistics, per-episode reproducibility, and actual policy RPC waiting."""
import random
import statistics
import threading
import unittest
from unittest.mock import patch

from wireless_rtc.cli import main
from wireless_rtc.network.transport import RPCClient
from wireless_rtc.server.delay import NormalResponseDelay
from wireless_rtc.server.service import make_server


class Policy:
    def describe(self):
        return {"horizon":8}

    def initialize(self, request):
        return {"actions":[[0.]]*8,"observation_tick":0}

    def infer(self, request):
        return {"actions":[[0.]]*8,"observation_tick":request["observation_tick"],"inference_ms":0.}


class ServerDelayTests(unittest.TestCase):
    def test_distribution_is_seeded_and_independent(self):
        global_state = random.getstate()
        first,second = NormalResponseDelay(),NormalResponseDelay()
        a,b = [first.sample() for _ in range(20000)],[second.sample() for _ in range(20000)]
        self.assertEqual(a,b)
        self.assertAlmostEqual(statistics.mean(a),150,delta=2)
        self.assertAlmostEqual(statistics.pstdev(a),50,delta=2)
        self.assertTrue(all(0 <= x <= 1000 for x in a))
        self.assertEqual(random.getstate(),global_state)

    def test_clipping_and_episode_reset(self):
        sampler = NormalResponseDelay()
        with patch.object(sampler.rng,"gauss",return_value=-20):
            self.assertEqual(sampler.sample(),0)
        with patch.object(sampler.rng,"gauss",return_value=2000):
            self.assertEqual(sampler.sample(),1000)
        sampler.reset(123)
        first = [sampler.sample() for _ in range(5)]
        sampler.reset(123)
        self.assertEqual(first,[sampler.sample() for _ in range(5)])
        sampler.reset(124)
        self.assertNotEqual(first,[sampler.sample() for _ in range(5)])

    def test_invalid_settings_fail_before_loading_checkpoint(self):
        for kwargs in [{"std_ms":-1},{"mean_ms":float("nan")},{"max_ms":0},{"max_ms":10001},{"seed":-1}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                NormalResponseDelay(**kwargs)
        with patch("sys.stderr"),self.assertRaises(SystemExit) as error:
            main(["serve","--checkpoint","missing.pkl","--delay","normal","--delay-std-ms","nan"])
        self.assertEqual(error.exception.code,2)

    def test_only_inference_is_delayed_and_initialization_replays_sequence(self):
        sampler = NormalResponseDelay(mean_ms=20,std_ms=3,max_ms=100)
        server = make_server("127.0.0.1",0,Policy(),response_delay=sampler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with RPCClient("127.0.0.1",server.server_address[1]) as client:
                info,_ = client.request("describe")
                self.assertEqual(info["response_delay"]["mean_ms"],20)
                sequences = []
                for _ in range(2):
                    initial,_ = client.request("initialize",seed=123,observation=[0])
                    self.assertNotIn("injected_delay_ms",initial)
                    samples = []
                    for tick in range(3):
                        probe,_ = client.request("probe",payload="test")
                        self.assertNotIn("injected_delay_ms",probe)
                        response,rtt = client.request("infer",observation_tick=tick)
                        samples.append(response["injected_delay_ms"])
                        self.assertGreaterEqual(response["delay_sleep_ms"],response["injected_delay_ms"]-.5)
                        self.assertGreaterEqual(response["server_ms"],response["delay_sleep_ms"])
                        self.assertGreaterEqual(rtt,response["injected_delay_ms"]-.5)
                    sequences.append(samples)
                self.assertEqual(*sequences)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)
