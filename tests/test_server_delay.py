"""Sample statistics, per-episode reproducibility, and actual policy RPC waiting."""
import random
import statistics
import threading
import unittest
from unittest.mock import patch

from wireless_rtc.cli import main
from wireless_rtc.network.transport import RPCClient
from wireless_rtc.server.delay import NormalResponseDelay, PatternResponseDelay
from wireless_rtc.server.service import make_server


class Policy:
    def describe(self):
        return {"horizon":8}

    def initialize(self, request):
        return {"actions":[[0.]]*8,"observation_tick":0}

    def infer(self, request):
        return {"actions":[[0.]]*8,"observation_tick":request["observation_tick"],"inference_ms":0.}


class ServerDelayTests(unittest.TestCase):
    def test_pattern_cycles_phases_and_resets_without_randomness(self):
        config = dict(repeat=True,phases=[dict(requests=2,values_ms=[10]),
                                        dict(requests=5,values_ms=[80,40,20]),dict(requests=1,values_ms=[200])])
        state = random.getstate()
        sampler = PatternResponseDelay(config)
        expected = [10.,10.,80.,40.,20.,80.,40.,200.]
        self.assertEqual([sampler.sample() for _ in range(10)],expected+[10.,10.])
        sampler.reset(123)
        self.assertEqual([sampler.sample() for _ in range(8)],expected)
        sampler.reset(999)
        self.assertEqual(sampler.sample(),10.)
        self.assertEqual(random.getstate(),state)
        config["phases"][0]["values_ms"][0] = 999
        self.assertEqual(sampler.describe()["pattern"]["phases"][0]["values_ms"],[10])
        held = PatternResponseDelay(dict(repeat=False,phases=[dict(requests=2,values_ms=[5,9])]))
        self.assertEqual([held.sample() for _ in range(4)],[5,9,9,9])

    def test_invalid_pattern_fails_before_checkpoint_loading(self):
        for config in [{},dict(phases=[]),dict(repeat="yes",phases=[{}]),
                       dict(phases=[dict(requests=0,values_ms=[10])]),
                       dict(phases=[dict(requests=True,values_ms=[10])]),
                       dict(phases=[dict(requests=1,values_ms=[float("nan")])]),
                       dict(phases=[dict(requests=1,values_ms=[-1])])]:
            with self.subTest(config=config),self.assertRaises(ValueError):
                PatternResponseDelay(config)
        with patch("sys.stderr"),self.assertRaises(SystemExit) as error:
            main(["serve","--checkpoint","missing.pkl","--delay","pattern","--delay-pattern","missing.json"])
        self.assertEqual(error.exception.code,2)

    def test_pattern_rpc_records_exact_sequence_and_resets_each_episode(self):
        sampler = PatternResponseDelay(dict(phases=[dict(requests=3,values_ms=[2,8,0])]))
        server = make_server("127.0.0.1",0,Policy(),response_delay=sampler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with RPCClient("127.0.0.1",server.server_address[1]) as client:
                info,_ = client.request("describe")
                self.assertEqual(info["response_delay"]["kind"],"deterministic_pattern")
                for seed in [123,999]:
                    client.request("initialize",seed=seed,observation=[0])
                    for expected in [2,8,0,2]:
                        client.request("probe",payload="unchanged")
                        response,_ = client.request("infer",observation_tick=0)
                        self.assertEqual(response["injected_delay_ms"],expected)
                        self.assertGreaterEqual(response["delay_sleep_ms"],expected-.5)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)

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
