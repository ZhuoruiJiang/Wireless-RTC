"""Real TCP tests for nonblocking simulation, action alignment and failed RPCs."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from wireless_rtc.client.live import ActionBuffer, run_live_episode, validate_actions, validate_config
from wireless_rtc.network.transport import RPCClient
from wireless_rtc.prediction.rolling import RollingPredictor
from wireless_rtc.server.service import make_server


class FakePolicy:
    def __init__(self, delay=0, bad_tick=False):
        self.delay, self.bad_tick = delay, bad_tick
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        time.sleep(self.delay)
        tick = request["observation_tick"]
        return dict(observation_tick=tick+int(self.bad_tick), inference_ms=self.delay*1000,
                    actions=[[(tick+1)*100+i] for i in range(len(request["previous_actions"]))])


@contextmanager
def connection(policy, timeout=1, response_delay=None):
    server = make_server("127.0.0.1",0,policy,response_delay=response_delay)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        with RPCClient("127.0.0.1",server.server_address[1],timeout) as client:
            yield client
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


class LiveTests(unittest.TestCase):
    def test_episode_averages_match_requests_with_injected_wait(self):
        from wireless_rtc.server.delay import NormalResponseDelay
        delay = NormalResponseDelay(mean_ms=3,std_ms=0,max_ms=3)
        with connection(FakePolicy(delay=.002),response_delay=delay) as client:
            result,_ = self.run_episode(client)
        successful = [row for row in self.requests if row["status"] == "ok"]
        self.assertTrue(successful)
        self.assertEqual(result["rtt_samples"],len(successful))
        for raw,average in [("actual_ms","average_rtt_ms"),
                            ("injected_delay_ms","average_injected_delay_ms"),
                            ("delay_sleep_ms","average_delay_sleep_ms")]:
            self.assertAlmostEqual(result[average],sum(row[raw] for row in successful)/len(successful))
        self.assertEqual(result["average_injected_delay_ms"],3.)
        self.assertGreater(result["average_delay_sleep_ms"],0.)
        self.assertAlmostEqual(result["prediction_mae_ms"],
                               sum(abs(row["predicted_ms"]-row["inference_ms"]) for row in successful)/len(successful))
        self.assertAlmostEqual(result["rtt_prediction_mae_ms"],
                               sum(abs(row["predicted_ms"]-row["actual_ms"]) for row in successful)/len(successful))
        self.assertEqual(result["prediction_samples"],len(successful))

    def test_rtt_uses_high_resolution_clock(self):
        with connection(FakePolicy(delay=.003)) as client, patch("time.monotonic",return_value=1000.):
            result,_ = self.run_episode(client,control_ms=5)
        self.assertEqual(result["status"],"ok")
        self.assertTrue(self.requests)
        self.assertTrue(all(r["actual_ms"] >= 2 for r in self.requests))

    def run_episode(self, client, horizon=8, max_steps=8, control_ms=10):
        self.actions, self.requests, self.ticks = [], [], []
        predictor = RollingPredictor(initial_ms=5)
        def step(state,action):
            self.actions.append(action[0])
            return [state+1],state+1,False,state+1 >= max_steps
        result = run_live_episode(client,[0],0,step,[[-10-i] for i in range(horizon)],
                                  predictor,control_ms,1,max_steps,0,"rtc",
                                  self.requests.append,self.ticks.append,action_ranges=[2.])
        return result,predictor

    def test_response_uses_current_suffix_instead_of_replaying_prefix(self):
        policy = FakePolicy(delay=.002)
        with connection(policy) as client:
            result,predictor = self.run_episode(client)
        self.assertEqual(result["status"],"ok")
        self.assertEqual(self.actions[0],-10)
        applied = [(action,row) for action,row in zip(self.actions,self.ticks) if action >= 100]
        self.assertTrue(applied)
        for action,row in applied:
            age = row["tick"]-row["action_origin_tick"]
            self.assertGreaterEqual(age,1)
            self.assertEqual(action,(row["action_origin_tick"]+1)*100+age)
        for request in policy.requests:
            self.assertLessEqual(request["predicted_steps"],request["overlap_steps"])
        completed = [r for r in self.requests if r["status"] == "ok" and r["arrival_tick"] is not None]
        self.assertEqual(len(predictor.history),len(completed))
        for i,row in enumerate(self.ticks):
            self.assertEqual(json.loads(row["action"]), [self.actions[i]])
            if row["chunk_switched"]:
                self.assertGreater(i,0)
                self.assertFalse(row["buffer_exhausted"])
                self.assertFalse(self.ticks[i-1]["buffer_exhausted"])
                self.assertAlmostEqual(row["boundary_action_jump"],abs(self.actions[i]-self.actions[i-1])/2)
        self.assertTrue(any(r["chunk_switched"] for r in self.ticks))
        for row in completed:
            self.assertEqual(row["buffer_deadline_tick"],row["observation_tick"]+row["overlap_steps"])
            self.assertEqual(row["deadline_missed"], row["arrival_tick"]>row["buffer_deadline_tick"] or row["stale_response"])

    def test_slow_policy_does_not_pause_physics_and_expired_chunks_are_discarded(self):
        with connection(FakePolicy(delay=.08)) as client:
            result,_ = self.run_episode(client,horizon=4,max_steps=22,control_ms=5)
        self.assertEqual(result["steps"],22)
        self.assertGreater(result["buffer_exhaustion_steps"],0)
        self.assertTrue(any(r.get("stale_response") for r in self.requests))
        self.assertTrue(all(a == 0 for a in self.actions[4:]))
        stale = next(r for r in self.requests if r.get("stale_response"))
        self.assertGreaterEqual(stale["elapsed_ticks"],4)
        self.assertEqual(stale["discard_reason"],"expired_chunk")
        self.assertTrue(stale["deadline_missed"])
        self.assertTrue(all(row["boundary_action_jump"] is None for row in self.ticks))

    def test_invalid_response_is_logged_and_never_updates_predictor(self):
        with connection(FakePolicy(bad_tick=True)) as client:
            result,predictor = self.run_episode(client)
        self.assertEqual(result["status"],"error")
        self.assertEqual(len(predictor.history),0)
        self.assertEqual(self.requests[0]["status"],"error")
        self.assertNotIn("actual_ms",self.requests[0])
        self.assertIn("tick mismatch",result["error"])
        self.assertEqual(result["rtt_samples"],0)
        self.assertIsNone(result["average_rtt_ms"])

    def test_timeout_is_censored_and_physics_advances_while_waiting(self):
        with connection(FakePolicy(delay=.08),timeout=.025) as client:
            result,predictor = self.run_episode(client,max_steps=100,control_ms=2)
        self.assertEqual(result["status"],"error")
        self.assertGreater(result["steps"],1)
        self.assertEqual(len(predictor.history),0)
        self.assertGreater(self.requests[0]["failed_elapsed_ms"],0)
        self.assertNotIn("actual_ms",self.requests[0])

    def test_end_of_episode_drains_pending_request_without_applying_it(self):
        with connection(FakePolicy(delay=.02)) as client:
            first,_ = self.run_episode(client,max_steps=1)
            first_id = self.requests[0]["request_id"]
            self.assertEqual(self.requests[0]["discard_reason"],"episode_ended")
            self.assertIsNone(self.requests[0]["arrival_tick"])
            self.assertFalse(self.requests[0]["deadline_evaluable"])
            self.assertIsNone(self.requests[0]["deadline_missed"])
            self.assertEqual(first["rtt_samples"],1)
            self.assertEqual(first["average_rtt_ms"],self.requests[0]["actual_ms"])
            second,_ = self.run_episode(client,max_steps=1)
            self.assertGreater(self.requests[0]["request_id"],first_id)
        self.assertEqual(first["status"],"ok")
        self.assertEqual(second["status"],"ok")
        self.assertEqual(self.actions,[-10])

    def test_action_buffer_aligns_and_exhausts(self):
        buffer = ActionBuffer([[10],[11],[12]],origin=5)
        self.assertEqual(buffer.aligned(6),([[11],[12],[0.]],2))
        self.assertEqual(buffer.aligned(10),([[0.],[0.],[0.]],0))
        self.assertEqual(buffer.action(8),([0.],True))

    def test_response_shape_and_finite_values_are_checked(self):
        for actions in [[[1]],[[float("nan")],[2]],[[1,2],[3,4]],[[True],[2]]]:
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                validate_actions(dict(observation_tick=0,actions=actions),0,2,1)

    def test_live_config_rejects_offline_delays(self):
        config = json.loads(Path("configs/rtc_remote.json").read_text())
        validate_config(config)
        config["delay"] = {"kind":"fixed","value_ms":150}
        with self.assertRaisesRegex(ValueError,"actual RPC delay"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
