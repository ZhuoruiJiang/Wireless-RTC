"""Paced simulator controller with one asynchronous policy RPC in flight."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import time

from wireless_rtc.client.schedule import delay_steps
from wireless_rtc.network.transport import RPCClient
from wireless_rtc.prediction.rolling import make_predictor
from wireless_rtc.control_modes import (CONTROL_MODES, PROTOCOL_VERSION, condition_name,
                                        inference_time, predictor_sample, predictors_for)


REQUEST_FIELDS = ["episode", "request_id", "observation_tick", "status", "actual_ms",
                  "predicted_ms", "predicted_steps", "applied_prefix_steps", "overlap_steps",
                  "prefix_clipped", "ideal_delay_steps", "arrival_tick", "elapsed_ticks",
                  "stale_response", "discard_reason", "server_ms", "inference_ms",
                  "injected_delay_ms", "delay_sleep_ms",
                  "predictor_basis", "predictor_observation_ms", "budget_exceeded",
                  "failed_elapsed_ms", "error"]
TICK_FIELDS = ["episode", "tick", "wall_time_ns", "action_origin_tick", "buffer_exhausted",
               "step_wall_ms", "deadline_overrun_ms"]


def validate_actions(response, tick, horizon, action_dim):
    if type(response.get("observation_tick")) is not int or response["observation_tick"] != tick:
        raise ValueError("Policy response observation tick mismatch")
    actions = response.get("actions")
    if not isinstance(actions, list) or len(actions) != horizon:
        raise ValueError("Policy response has invalid action horizon")
    for action in actions:
        if not isinstance(action, list) or len(action) != action_dim or any(
                isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) for x in action):
            raise ValueError("Policy response has invalid action dimensions or nonfinite values")
    return actions


class ActionBuffer:
    def __init__(self, actions, origin=0):
        self.actions, self.origin = actions, origin
        self.horizon, self.action_dim = len(actions), len(actions[0])

    def aligned(self, tick):
        offset = tick - self.origin
        valid = max(0,self.horizon-offset)
        return self.actions[offset:] + [[0.]*self.action_dim for _ in range(self.horizon-valid)], valid

    def action(self, tick):
        offset = tick - self.origin
        if 0 <= offset < self.horizon:
            return self.actions[offset], False
        return [0.]*self.action_dim, True


def run_live_episode(client, observation, state, step, initial_actions, predictor,
                     control_ms, min_execution_steps, max_steps, episode, method,
                     write_request, write_tick, viewer=None, initial_inference_ms=None):
    """`step` returns (observation-list, state, success, done).

    Only this thread advances physics. RPC serialization/decoding runs on a single
    worker. Response chunks retain their observation-tick origin; expired indices
    are never replayed. No injected delay or future latency is available here.
    """
    if method not in CONTROL_MODES:
        raise ValueError("Unknown control method")
    if method == "naive":
        predictor = None
    elif predictor is None:
        raise ValueError("RTC methods require a predictor")
    if method == "rtc" and initial_inference_ms is not None:
        predictor.update(inference_time(initial_inference_ms))
    buffer = ActionBuffer(initial_actions)
    horizon, action_dim = buffer.horizon, buffer.action_dim
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-rpc")
    pending = None
    tick = next_request_tick = exhausted = overruns = 0
    solved, error, last_ms = False, "", None
    last_injected_ms = None
    last_inference_ms = None
    completed_requests = stale_requests = 0

    def request(fields):
        # Include serialization, network, inference, decoding and validation in RTT.
        try:
            response, _ = client.request("infer", **fields)
            validate_actions(response, fields["observation_tick"], horizon, action_dim)
            if method == "rtc":
                inference_time(response.get("inference_ms"))
            return response, "", time.perf_counter()
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}", time.perf_counter()

    def receive(arrival_tick):
        nonlocal buffer, pending, completed_requests, stale_requests, last_ms, last_injected_ms, last_inference_ms
        response, failure, received_at = pending["future"].result()
        elapsed_ms = (received_at-pending["submitted"])*1000
        row = dict(pending["row"])
        row.update(status="error" if failure else "ok", arrival_tick=arrival_tick,
                   error=failure, discard_reason="episode_ended" if arrival_tick is None else "")
        if failure:
            row["failed_elapsed_ms"] = elapsed_ms  # Censored; never feed this to the predictor.
        else:
            age = arrival_tick-row["observation_tick"] if arrival_tick is not None else None
            stale = age >= horizon if age is not None else None
            row.update(request_id=response["request_id"], actual_ms=elapsed_ms,
                       ideal_delay_steps=delay_steps(elapsed_ms,control_ms), elapsed_ticks=age,
                       stale_response=stale, server_ms=response.get("server_ms"),
                       inference_ms=response.get("inference_ms"), injected_delay_ms=response.get("injected_delay_ms"),
                       delay_sleep_ms=response.get("delay_sleep_ms"))
            sample = predictor_sample(method,response.get("inference_ms"),elapsed_ms)
            row.update(predictor_observation_ms=sample,
                       budget_exceeded=elapsed_ms>row["predicted_ms"] if predictor is not None else None)
            completed_requests += 1
            if arrival_tick is not None:
                if predictor is not None:
                    predictor.update(sample)
                last_ms = elapsed_ms
                last_injected_ms = response.get("injected_delay_ms")
                last_inference_ms = response.get("inference_ms")
                if stale:
                    stale_requests += 1
                    row["discard_reason"] = "expired_chunk"
                else:
                    buffer = ActionBuffer(response["actions"],row["observation_tick"])
        write_request(row)
        pending = None
        return failure

    try:
        while tick < max_steps:
            boundary = time.perf_counter()
            if pending is not None and pending["future"].done():
                error = receive(tick)
                if error:
                    break
            if pending is None and tick >= next_request_tick:
                prediction = predictor.predict() if predictor is not None else None
                raw_d = delay_steps(prediction,control_ms) if prediction is not None else 0
                previous, valid = buffer.aligned(tick)
                row = dict(episode=episode, request_id=client.counter, observation_tick=tick, predicted_ms=prediction,
                           predicted_steps=raw_d, applied_prefix_steps=min(raw_d,valid),
                           overlap_steps=valid, prefix_clipped=raw_d>valid,
                           predictor_basis=CONTROL_MODES[method])
                fields = dict(observation=observation, previous_actions=previous,
                              predicted_steps=min(raw_d,valid), overlap_steps=valid,
                              observation_tick=tick, method=method)
                submitted = time.perf_counter()
                pending = dict(row=row, submitted=submitted, future=executor.submit(request,fields))
                next_request_tick = tick + min_execution_steps
            action, empty = buffer.action(tick)
            started = time.perf_counter()
            observation, state, success, done = step(state,action)
            step_ms = (time.perf_counter()-started)*1000
            solved = solved or success
            exhausted += int(empty)
            if viewer is not None:
                viewer.update(state, phase="Live remote policy", tick=tick+1,
                              simulation_ms=(tick+1)*control_ms, success=solved,
                              delay_kind="measured", predicted_ms=predictor.predict() if predictor is not None else None, actual_ms=last_ms,
                              inference_ms=last_inference_ms,predictor_basis=CONTROL_MODES[method],
                              injected_delay_ms=last_injected_ms, buffer_exhaustion_steps=exhausted)
            overrun_ms = max(0,(time.perf_counter()-boundary)*1000-control_ms)
            overruns += int(overrun_ms>0)
            write_tick(dict(episode=episode, tick=tick, wall_time_ns=time.time_ns(),
                            action_origin_tick=buffer.origin, buffer_exhausted=empty,
                            step_wall_ms=step_ms, deadline_overrun_ms=overrun_ms))
            tick += 1
            if done or tick >= max_steps:
                break
            # Never execute a burst of catch-up physics steps after an overrun.
            time.sleep(max(0,boundary+control_ms/1000-time.perf_counter()))
        # The episode has ended: drain its request before resetting the server RNG.
        # No returned actions can leak into the next episode.
        if pending is not None:
            error = receive(None) or error
    finally:
        if pending is not None:  # Interrupted/failed while RPC is still running.
            client.close()
        executor.shutdown(wait=True, cancel_futures=True)
    return dict(episode=episode, status="error" if error else "ok", error=error,
                success=bool(solved), steps=tick, buffer_exhaustion_steps=exhausted,
                deadline_overrun_steps=overruns, completed_requests=completed_requests,
                stale_requests=stale_requests)


def validate_config(config):
    if "delay" in config:
        raise ValueError("rtc-live measures actual RPC delay; use configs/rtc_remote.json without injected delay")
    if (type(config["episodes"]) is not int or config["episodes"] < 1
            or type(config["min_execution_steps"]) is not int or config["min_execution_steps"] < 1):
        raise ValueError("episodes and min_execution_steps must be positive integers")
    if not math.isfinite(config["control_ms"]) or config["control_ms"] <= 0:
        raise ValueError("control_ms must be finite and positive")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if not config["methods"] or any(m not in CONTROL_MODES for m in config["methods"]):
        raise ValueError("methods must contain naive, rtc and/or wireless_rtc")
    if len(set(config["methods"])) != len(config["methods"]) or not config["predictors"]:
        raise ValueError("Methods must be unique and predictors nonempty")
    for name, settings in config["predictors"].items():
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Predictor names must use letters, numbers, underscores or hyphens")
        make_predictor(settings)


def run_remote(config, host, port, output, metadata, timeout=120, viewer=None):
    validate_config(config)
    from wireless_rtc.simulation.kinetix import KinetixSimulator
    sim = KinetixSimulator(config["level"],config["seed"])
    jax,jnp,np = sim.jax,sim.jnp,sim.np
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest_path = output/"manifest.json"
    metadata.update(kind="pretrained_kinetix_live_rpc", status="running", config=config,
                    host=host, port=port, timeout=timeout, client_devices=[str(d) for d in jax.devices()],
                    viewer=dict(enabled=viewer is not None, fps=viewer.fps if viewer else None),
                    timing="Paced wall-clock simulation; RTT from submission through response validation",
                    fallback="Zero action while physics advances when the action buffer is exhausted")
    metadata["control_modes"] = CONTROL_MODES.copy()
    def save_manifest():
        manifest_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    save_manifest()
    if viewer:
        viewer.set_renderer(sim.make_frame_renderer())
    summaries = []
    try:
        with RPCClient(host,port,timeout) as client:
            description,_ = client.request("describe")
            if description.get("protocol_version") != PROTOCOL_VERSION or description.get("control_modes") != CONTROL_MODES:
                raise ValueError("Three-mode protocol v2 required: update and restart the policy server and client")
            if (description.get("level_sha256") != sim.level_sha256
                    or description.get("observation_dim") != sim.observation_dim
                    or description.get("action_dim") != sim.action_dim):
                raise ValueError("Client simulator and policy server are incompatible; check revisions and level")
            horizon = description["horizon"]
            if type(horizon) is not int or not 1 <= config["min_execution_steps"] < horizon:
                raise ValueError("Require 1 <= min_execution_steps < policy horizon")
            metadata["server"] = description
            save_manifest()
            for method in config["methods"]:
                for name,settings in predictors_for(method,config):
                    directory = output/condition_name(method,name)
                    directory.mkdir()
                    episodes = []
                    with (directory/"requests.csv").open("w",newline="",encoding="utf-8") as rf, (
                            directory/"ticks.csv").open("w",newline="",encoding="utf-8") as tf:
                        requests = csv.DictWriter(rf,fieldnames=REQUEST_FIELDS)
                        ticks = csv.DictWriter(tf,fieldnames=TICK_FIELDS)
                        requests.writeheader(); ticks.writeheader()
                        def write_request(row):
                            requests.writerow(row); rf.flush()
                        def write_tick(row):
                            ticks.writerow(row); tf.flush()
                        for episode in range(config["episodes"]):
                            if viewer:
                                viewer.update(phase="Warming physics / policy before episode", method=method,
                                              predictor=name, episode=episode+1, episodes=config["episodes"],
                                              tick=0, success=False, actual_ms=None, delay_kind="measured",
                                              buffer_exhaustion_steps=0,injected_delay_ms=None,
                                              predictor_basis=CONTROL_MODES[method],predicted_ms=None,inference_ms=None)
                            seed = (config["seed"]+episode) % 2**32
                            key = jax.random.key(seed)
                            obs,state = sim.reset(key)
                            # Compile/synchronize physics without consuming episode steps or RNG.
                            warm = sim.step(key,state,jnp.zeros(sim.action_dim))
                            jax.block_until_ready(warm)
                            initial,_ = client.request("initialize", observation=np.asarray(obs).tolist(),seed=seed)
                            actions = validate_actions(initial,0,horizon,sim.action_dim)
                            calibration_ms = inference_time(initial.get("calibration_inference_ms")) if method == "rtc" else None
                            env_key = jax.random.fold_in(key,1)
                            def step(state,action):
                                nonlocal env_key
                                env_key,step_key = jax.random.split(env_key)
                                obs,state,reward,done,info = sim.step(step_key,state,jnp.asarray(action))
                                return (np.asarray(obs).tolist(),state,
                                        bool(info.get("success",reward>0)),bool(done))
                            result = run_live_episode(client,np.asarray(obs).tolist(),state,step,actions,
                                                      make_predictor(settings) if settings is not None else None,config["control_ms"],
                                                      config["min_execution_steps"],int(sim.params.max_timesteps),
                                                      episode,method,write_request,write_tick,viewer,
                                                      initial_inference_ms=calibration_ms)
                            result["calibration_inference_ms"] = calibration_ms
                            episodes.append(result)
                            with (directory/"episodes.csv").open("w",newline="",encoding="utf-8") as ef:
                                writer = csv.DictWriter(ef,fieldnames=list(result))
                                writer.writeheader(); writer.writerows(episodes)
                            print(json.dumps(dict(method=method,predictor=name,**result)),flush=True)
                            if result["status"] != "ok":
                                raise RuntimeError(result["error"])
                    summary = dict(method=method,predictor=name,predictor_basis=CONTROL_MODES[method],episodes=len(episodes),
                                   solve_rate=sum(e["success"] for e in episodes)/len(episodes),
                                   buffer_exhaustion_steps=sum(e["buffer_exhaustion_steps"] for e in episodes),
                                   deadline_overrun_steps=sum(e["deadline_overrun_steps"] for e in episodes))
                    summaries.append(summary)
                    (directory/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        (output/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
        metadata["status"] = "completed"
        if viewer:
            viewer.update(phase="Completed. Local viewer retained; Ctrl+C closes it.")
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "error",
                        error=f"{type(exc).__name__}: {exc}")
        if viewer:
            viewer.update(phase=metadata["status"],error=metadata["error"])
        raise
    finally:
        save_manifest()
