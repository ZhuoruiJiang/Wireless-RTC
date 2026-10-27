import io
import math
from pathlib import Path
import tempfile
import threading
import unittest

from wireless_rtc.analysis.evaluate import evaluate, read_trace, save_evaluation
from wireless_rtc.client.schedule import delay_steps, execute_interval
from wireless_rtc.network.delays import generate_delays
from wireless_rtc.network.transport import RPCClient
from wireless_rtc.prediction.rolling import RollingPredictor
from wireless_rtc.protocol.messages import receive_message, send_message
from wireless_rtc.server.service import make_server


class PredictorTests(unittest.TestCase):
    def test_prediction_has_no_current_target(self):
        rows, _ = evaluate([40,40,200,40], {"kind":"max","window":2,"initial_ms":10}, warmup=0)
        self.assertEqual([r["predicted_ms"] for r in rows], [10,40,40,200])

    def test_rolling_window_expires_spike(self):
        p = RollingPredictor(window=2)
        for x in [200,40,40]:
            p.update(x)
        self.assertEqual(p.predict(),40)

    def test_invalid_measurements(self):
        for x in [-1,float("nan"),float("inf")]:
            with self.assertRaises(ValueError):
                RollingPredictor().update(x)

    def test_fixed_delay_converges(self):
        _, m = evaluate([60]*100, {"kind":"quantile"}, warmup=32)
        self.assertEqual(m["mae_ms"],0)
        self.assertEqual(m["exceedance_rate"],0)

    def test_warmup_excluded(self):
        rows, m = evaluate([60]*5, {"initial_ms":0},warmup=1)
        self.assertFalse(rows[0]["scored"])
        self.assertEqual(m["samples"],4)
        self.assertEqual(m["mae_ms"],0)

    def test_seeded_trace_is_shared(self):
        c={"kind":"uniform","low_ms":20,"high_ms":80}
        self.assertEqual(generate_delays(c,100,42),generate_delays(c,100,42))
        self.assertNotEqual(generate_delays(c,100,42),generate_delays(c,100,43))

    def test_censored_trace_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"trace.csv"
            p.write_text("status,actual_ms\nok,20\nerror,\n")
            with self.assertRaises(ValueError):
                read_trace(p)

    def test_results_round_trip(self):
        rows,m=evaluate([60]*5,{},warmup=1)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"result"
            save_evaluation(p,rows,m,{"kind":"test"})
            self.assertEqual(read_trace(p/"predictions.csv"),[60]*5)
            self.assertIn("<svg",(p/"prediction.svg").read_text())
            with self.assertRaises(FileExistsError):
                save_evaluation(p,rows,m,{})


class ScheduleTests(unittest.TestCase):
    def test_elapsed_prefix_is_not_replayed(self):
        old=[[10],[11],[12],[13]]
        new=[[20],[21],[22],[23]]
        actions,misses=execute_interval(old,new,2,4,4)
        self.assertEqual(actions,[[10],[11],[22],[23]])
        self.assertEqual(misses,0)

    def test_underestimate_does_not_imply_buffer_exhaustion(self):
        # Predictor is deliberately irrelevant to action-arrival scheduling.
        old=[[10],[11],[12],[13]]
        actions,misses=execute_interval(old,[[20],[21],[22],[23]],3,4,4)
        self.assertEqual(actions,[[10],[11],[12],[23]])
        self.assertEqual(misses,0)

    def test_exhaustion_uses_zero_not_padded_actions(self):
        actions,misses=execute_interval([[10],[99],[99],[99]],[[20],[21],[22],[23]],3,4,1)
        self.assertEqual(actions,[[10],[0.],[0.],[23]])
        self.assertEqual(misses,2)

    def test_grid_rounding(self):
        self.assertEqual(delay_steps(40,20),2)
        self.assertEqual(delay_steps(40.1,20),3)
        self.assertEqual(delay_steps(0,20),0)


class ProtocolTests(unittest.TestCase):
    def test_persistent_connection_and_timing(self):
        server=make_server("127.0.0.1",0)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with RPCClient("127.0.0.1",server.server_address[1]) as client:
                for i in range(2):
                    response,rtt=client.request("probe",hold_ms=15,payload="abc")
                    self.assertEqual(response["request_id"],i)
                    self.assertEqual(response["payload"],"abc")
                    self.assertGreaterEqual(rtt,14)
                    self.assertGreaterEqual(response["server_ms"],14)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_nonfinite_and_oversized_frames_rejected(self):
        with self.assertRaises(ValueError):
            receive_message(io.BytesIO(b'{"delay":NaN}\n'))
        with self.assertRaises(ValueError):
            send_message(io.BytesIO(),{"x":float("nan")})


if __name__=="__main__":
    unittest.main()
