from collections import deque
import math


def quantile(values, q):
    """Linear-interpolated empirical quantile."""
    if not values or not 0 <= q <= 1:
        raise ValueError("Quantile requires values and q in [0,1]")
    xs = sorted(values)
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


class RollingPredictor:
    def __init__(self, kind="max", window=32, quantile_level=.95,
                 initial_ms=60., margin_ms=0.):
        if kind not in {"max", "quantile", "last"}:
            raise ValueError(f"Unknown predictor: {kind}")
        if window < 1 or not 0 <= quantile_level <= 1:
            raise ValueError("Invalid window or quantile")
        if not all(math.isfinite(x) and x >= 0 for x in [initial_ms, margin_ms]):
            raise ValueError("Initial estimate and margin must be finite and nonnegative")
        self.kind, self.q = kind, quantile_level
        self.initial_ms, self.margin_ms = initial_ms, margin_ms
        self.history = deque(maxlen=window)

    def predict(self):
        if not self.history:
            return self.initial_ms + self.margin_ms
        estimate = self.history[-1] if self.kind == "last" else quantile(
            self.history, 1 if self.kind == "max" else self.q)
        return estimate + self.margin_ms

    def update(self, measured_ms):
        if not math.isfinite(measured_ms) or measured_ms < 0:
            raise ValueError("Delay must be finite and nonnegative")
        self.history.append(float(measured_ms))


def make_predictor(config):
    return RollingPredictor(**config)
