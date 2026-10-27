import math


def delay_steps(delay_ms, control_ms):
    """Conservative grid: response usable at the next controller tick."""
    if not math.isfinite(delay_ms) or delay_ms < 0 or not math.isfinite(control_ms) or control_ms <= 0:
        raise ValueError("Invalid delay or control interval")
    return math.ceil(delay_ms / control_ms)


def execute_interval(previous, new, actual_delay, horizon, valid_previous):
    """Both chunks start at observation time; never replay elapsed new actions."""
    if horizon < 1 or actual_delay < 0 or not 0 <= valid_previous <= len(previous):
        raise ValueError("Invalid scheduling inputs")
    actions, exhausted = [], 0
    for tick in range(horizon):
        if tick >= actual_delay and tick < len(new):
            actions.append(new[tick])
        elif tick < actual_delay and tick < valid_previous:
            actions.append(previous[tick])
        else:
            actions.append([0.] * len(previous[0]))
            exhausted += 1
    return actions, exhausted
