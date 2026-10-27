"""Run a temporary loopback service, collect a trace, and shut it down."""
import argparse
import threading
from wireless_rtc.network.probe import run_probe
from wireless_rtc.server.service import make_server

parser=argparse.ArgumentParser()
parser.add_argument("--output",required=True)
parser.add_argument("--samples",type=int,default=50)
args=parser.parse_args()
server=make_server("127.0.0.1",0)
thread=threading.Thread(target=server.serve_forever,daemon=True)
thread.start()
try:
    print(run_probe("127.0.0.1",server.server_address[1],args.output,
                    samples=args.samples,interval_ms=0,
                    delay_config={"kind":"uniform","low_ms":10,"high_ms":70}))
finally:
    server.shutdown()
    server.server_close()
    thread.join()
