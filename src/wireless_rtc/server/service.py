import socketserver
import time
from wireless_rtc.protocol.messages import receive_message, send_message


def make_server(host="127.0.0.1", port=8765, policy=None, response_delay=None):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(300)
            while True:
                try:
                    request = receive_message(self.rfile)
                except (EOFError, OSError, ValueError):
                    return
                start = time.perf_counter()
                try:
                    operation = request.get("operation")
                    if operation == "probe":
                        hold = float(request.get("hold_ms", 0))
                        if not 0 <= hold <= 10000:
                            raise ValueError("hold_ms must be in [0,10000]")
                        time.sleep(hold / 1000)
                        result = {"payload": request.get("payload", "")}
                    elif operation == "infer" and policy is not None:
                        result = policy.infer(request)
                        injected_ms = response_delay.sample() if response_delay is not None else 0.
                        wait_started = time.perf_counter()
                        if injected_ms:
                            time.sleep(injected_ms/1000)
                        result.update(injected_delay_ms=injected_ms,
                                      delay_sleep_ms=(time.perf_counter()-wait_started)*1000 if injected_ms else 0.)
                    elif operation == "describe" and policy is not None:
                        result = policy.describe()
                        result["response_delay"] = response_delay.describe() if response_delay is not None else {"kind":"none"}
                    elif operation == "select_task" and policy is not None:
                        task = request["task"]
                        if hasattr(policy,"select_task"):
                            result = policy.select_task(task)
                        else:
                            result = policy.describe()
                            if result.get("task") != task:
                                raise ValueError("Single-policy server cannot select this task; start with --checkpoint-dir")
                        result["response_delay"] = response_delay.describe() if response_delay is not None else {"kind":"none"}
                    elif operation == "initialize" and policy is not None:
                        result = policy.initialize(request)
                        if response_delay is not None:
                            response_delay.reset(request["seed"])
                    else:
                        raise ValueError("Unsupported operation or no policy loaded")
                    response = dict(request_id=request["request_id"], status="ok",
                                    server_ms=(time.perf_counter()-start)*1000, **result)
                except (ValueError, KeyError, TypeError) as exc:
                    response = dict(request_id=request.get("request_id"), status="error", error=str(exc))
                try:
                    send_message(self.wfile, response)
                except OSError:
                    return
    # Serial service avoids concurrent policy state/RNG mutation. One experimental client.
    class Server(socketserver.TCPServer):
        allow_reuse_address = True
    return Server((host, port), Handler)


def serve(host, port, policy=None, response_delay=None):
    with make_server(host, port, policy, response_delay) as server:
        print(f"Listening on {host}:{server.server_address[1]}", flush=True)
        if response_delay is not None:
            print(f"Injected response delay: {response_delay.describe()}",flush=True)
        server.serve_forever()
