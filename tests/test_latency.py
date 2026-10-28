import unittest

from wireless_rtc.analysis.latency import LatencyStats, combine_latency
from wireless_rtc.analysis.tasks import aggregate_tasks


class LatencyTests(unittest.TestCase):
    def test_mae_uses_method_target_and_reports_common_rtt_error(self):
        rtc, wireless = LatencyStats(), LatencyStats()
        for predicted, inference, rtt in [(10.,4.,40.),(0.,8.,20.)]:
            row = dict(status="ok",predicted_ms=predicted,inference_ms=inference,actual_ms=rtt)
            rtc.observe(dict(row,predictor_basis="inference_ms"))
            wireless.observe(dict(row,predictor_basis="response_rtt_ms"))
        self.assertEqual(rtc.summary()["prediction_mae_ms"],7.)
        self.assertEqual(rtc.summary()["rtt_prediction_mae_ms"],25.)
        self.assertEqual(wireless.summary()["prediction_mae_ms"],25.)
        self.assertEqual(wireless.summary()["rtt_prediction_mae_ms"],25.)
        self.assertEqual(rtc.summary()["prediction_samples"],2)

    def test_mae_excludes_failed_missing_invalid_and_naive_predictions(self):
        stats = LatencyStats()
        for changes in [dict(status="error"),dict(predicted_ms=None),dict(predicted_ms=float("inf")),
                        dict(predicted_ms=True),dict(predictor_basis="none")]:
            row = dict(status="ok",predicted_ms=10.,inference_ms=4.,actual_ms=40.,predictor_basis="inference_ms")
            row.update(changes)
            stats.observe(row)
        self.assertIsNone(stats.summary()["prediction_mae_ms"])
        self.assertIsNone(stats.summary()["rtt_prediction_mae_ms"])
        self.assertEqual(stats.summary()["prediction_samples"],0)
        # Missing native target does not erase an available RTT error.
        stats.observe(dict(status="ok",predicted_ms=10.,actual_ms=40.,predictor_basis="inference_ms"))
        self.assertEqual(stats.summary()["rtt_prediction_mae_ms"],30.)
        self.assertIsNone(stats.summary()["prediction_mae_ms"])

    def test_only_successful_measured_values_contribute(self):
        stats = LatencyStats()
        stats.observe(dict(status="ok", actual_ms=10., injected_delay_ms=0., delay_sleep_ms=0.))
        stats.observe(dict(status="ok", actual_ms=30., injected_delay_ms=20., delay_sleep_ms=22.,
                           discard_reason="expired_chunk"))
        stats.observe(dict(status="ok", actual_ms=50., discard_reason="episode_ended"))
        stats.observe(dict(status="error", actual_ms=900., injected_delay_ms=800., failed_elapsed_ms=900.))
        stats.observe(dict(status="ok", actual_ms=float("nan"), injected_delay_ms=-1., delay_sleep_ms=True))
        summary = stats.summary()
        self.assertEqual(summary["average_rtt_ms"], 30.)
        self.assertEqual(summary["rtt_samples"], 3)
        self.assertEqual(summary["average_injected_delay_ms"], 10.)
        self.assertEqual(summary["injected_delay_samples"], 2)
        self.assertEqual(summary["average_delay_sleep_ms"], 11.)

    def test_missing_samples_have_no_average(self):
        stats = LatencyStats().summary()
        self.assertIsNone(stats["average_rtt_ms"])
        self.assertEqual(stats["rtt_samples"], 0)
        self.assertEqual(combine_latency([stats]), stats)

    def test_task_latency_is_request_weighted_while_score_is_equal_weighted(self):
        rows = []
        for task, count, rtt, wait, score in [("a",1,100.,50.,0.),("b",9,20.,0.,1.)]:
            stats = LatencyStats()
            for _ in range(count):
                stats.observe(dict(status="ok",actual_ms=rtt,injected_delay_ms=wait,
                                   predicted_ms=0.,inference_ms=rtt,predictor_basis="inference_ms"))
            rows.append(dict(task=task, method="rtc", predictor="p95", predictor_basis="inference_ms",
                             episodes=1, solve_rate=score, buffer_exhaustion_steps=0,
                             deadline_overrun_steps=0, **stats.summary()))
        overall = aggregate_tasks(rows,["a","b"],[("rtc","p95")])["overall"][0]
        self.assertEqual(overall["average_solve_rate"], .5)
        self.assertEqual(overall["average_rtt_ms"], 28.)
        self.assertEqual(overall["average_injected_delay_ms"], 5.)
        self.assertEqual(overall["rtt_samples"], 10)
        self.assertEqual(overall["prediction_mae_ms"],28.)
        self.assertEqual(overall["prediction_samples"],10)
