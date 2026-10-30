"""Explicit experimental control modes shared by live and offline runners."""
import math


CONTROL_MODES = {"naive":"none", "rtc":"inference_ms", "wireless_rtc":"response_rtt_ms"}
PROTOCOL_VERSION = 3


def inference_time(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
        raise ValueError("RTC baseline requires a finite nonnegative server inference timing")
    return float(value)


def predictor_sample(method, inference_ms, response_ms):
    if method == "rtc":
        return inference_time(inference_ms)
    if method == "wireless_rtc":
        return response_ms
    if method == "naive":
        return None
    raise ValueError(f"Unknown control method: {method}")


def predictors_for(method,config):
    if method not in CONTROL_MODES:
        raise ValueError(f"Unknown control method: {method}")
    if method == "naive":
        return [("none", None)]
    selected = config.get("method_predictors", {}).get(method)
    if selected is None:
        return config["predictors"].items()
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise ValueError("method_predictors must select a nonempty unique list")
    if any(name not in config["predictors"] for name in selected):
        raise ValueError("method_predictors references an unknown predictor")
    return [(name, config["predictors"][name]) for name in selected]


def condition_name(method,predictor):
    return "naive" if method == "naive" else f"{method}-{predictor}"
