import socketserver
import time
from wireless_rtc.protocol.messages import receive_message, send_message


def make_server(host="127.0.0.1", port=8765, policy=None):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(60)
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
                    elif operation == "describe" and policy is not None:
                        result = policy.describe()
                    elif operation == "initialize" and policy is not None:
                        result = policy.initialize(request)
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


def serve(host, port, policy=None):
    with make_server(host, port, policy) as server:
        print(f"Listening on {host}:{server.server_address[1]}", flush=True)
        server.serve_forever()
