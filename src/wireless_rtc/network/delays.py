import math
import random


def generate_delays(config, count, seed):
    """Use independent RNG, never the policy or predictor RNG."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    kind = config["kind"]
    if kind == "fixed":
        values = [float(config["value_ms"])] * count
    elif kind == "uniform":
        low, high = config["low_ms"], config["high_ms"]
        if low > high:
            raise ValueError("low_ms must not exceed high_ms")
        values = [rng.uniform(low, high) for _ in range(count)]
    elif kind == "bursty":
        enter, leave = config.get("p_enter", .03), config.get("p_leave", .15)
        if not 0 <= enter <= 1 or not 0 <= leave <= 1:
            raise ValueError("Transition probabilities must be in [0,1]")
        burst, values = False, []
        for _ in range(count):
            burst = not (rng.random() < leave) if burst else rng.random() < enter
            center = config.get("burst_ms", 120) if burst else config.get("base_ms", 35)
            values.append(max(0, center + rng.gauss(0, config.get("jitter_ms", 5))))
    else:
        raise ValueError(f"Unknown delay source: {kind}")
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Delays must be finite and nonnegative")
    return values
