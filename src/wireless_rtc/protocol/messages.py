import json

MAX_MESSAGE_BYTES = 4 * 1024 * 1024


def send_message(stream, message):
    data = json.dumps(message, allow_nan=False).encode("utf-8") + b"\n"
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Message too large")
    stream.write(data)
    stream.flush()


def receive_message(stream):
    raw = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not raw:
        raise EOFError("Peer closed connection")
    if len(raw) > MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
        raise ValueError("Invalid or oversized frame")
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    value = json.loads(raw, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value
