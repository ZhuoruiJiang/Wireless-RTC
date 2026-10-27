import socket
import time
from wireless_rtc.protocol.messages import receive_message, send_message


class RPCClient:
    def __init__(self, host, port, timeout=10):
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.stream = self.socket.makefile("rwb")
        self.counter = 0

    def request(self, operation, **fields):
        request_id = self.counter
        self.counter += 1
        start = time.perf_counter()
        send_message(self.stream, dict(request_id=request_id, operation=operation, **fields))
        response = receive_message(self.stream)
        elapsed_ms = (time.perf_counter()-start)*1000
        if response.get("request_id") != request_id:
            raise ValueError("Response/request mismatch")
        if response.get("status") != "ok":
            raise ValueError(response.get("error", "Server error"))
        return response, elapsed_ms

    def close(self):
        # Wake a background reader first; file.close alone may wait on its lock.
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.stream.close()
        finally:
            self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
