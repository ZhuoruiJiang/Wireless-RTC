"""Shared upstream physics/policy adapters and the offline logical-time runner.

The paced live RPC runner lives in wireless_rtc.client.live.
"""
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import pickle
import sys
import time

from wireless_rtc.client.schedule import delay_steps, execute_interval
from wireless_rtc.network.delays import generate_delays
from wireless_rtc.prediction.rolling import make_predictor
from wireless_rtc.control_modes import CONTROL_MODES, PROTOCOL_VERSION, condition_name, predictors_for, predictor_sample


UPSTREAM = Path(__file__).resolve().parents[3] / "vendor" / "rtc-kinetix"


def dependencies():
    sys.path.insert(0, str(UPSTREAM / "src"))
    sys.path.insert(0, str(UPSTREAM / "third_party" / "kinetix"))
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import numpy as np
    from kinetix.environment import env as kenv
    from kinetix.util import saving
    model = importlib.import_module("model")
    return jax, jnp, nnx, np, kenv, saving, model


class KinetixSimulator:
    """Client-side physics and rendering; no checkpoint or policy is loaded."""

    def __init__(self, level="worlds/l/grasp_easy.json", seed=0):
        sys.path.insert(0, str(UPSTREAM / "third_party" / "kinetix"))
        import jax
        import jax.numpy as jnp
        import numpy as np
        from kinetix.environment import env as kenv
        from kinetix.util import saving
        self.jax, self.jnp, self.np = jax, jnp, np
        jax, jnp = self.jax, self.jnp
        self.level_name = level
        self.level_sha256 = hashlib.sha256((UPSTREAM / level).read_bytes()).hexdigest()
        self.level, static, self.params = saving.load_from_json_file(str(UPSTREAM / level))
        self.static = static.replace(screen_dim=(512,512))
        self.env = kenv.make_kinetix_env_from_name(
            "Kinetix-Symbolic-Continuous-v1", static_env_params=self.static)
        obs, _ = self.env.reset_to_level(jax.random.key(seed), self.level, self.params)
        self.observation_dim = obs.shape[-1]
        self.action_dim = self.env.action_space(self.params).shape[0]
        self.reset = jax.jit(lambda key: self.env.reset_to_level(key,self.level,self.params))
        def step(key,state,action):
            key1,key2 = jax.random.split(key)
            noisy = action + jax.random.normal(key1,action.shape)*.1
            return self.env.step_env(key2,state,noisy,self.params)
        self.step = jax.jit(step)

    def make_frame_renderer(self):
        from wireless_rtc.simulation.rendering import KinetixFrameRenderer
        return KinetixFrameRenderer(self)


class KinetixPolicy(KinetixSimulator):
    def __init__(self, checkpoint, level="worlds/l/grasp_easy.json", seed=0):
        super().__init__(level, seed)
        _, _, nnx, _, _, _, model = dependencies()
        jax, jnp = self.jax, self.jnp
        self.checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.policy = model.FlowPolicy(obs_dim=self.observation_dim, action_dim=self.action_dim,
                                       config=model.ModelConfig(), rngs=nnx.Rngs(seed))
        # Checkpoints are upstream Python pickle files: load trusted local files only.
        with Path(checkpoint).open("rb") as f:
            state_dict = pickle.load(f)
        graph, state = nnx.split(self.policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graph, state)
        self.policy = policy
        self.horizon = policy.action_chunk_size
        self.parameter_count = sum(int(x.size) for x in jax.tree.leaves(nnx.state(policy,nnx.Param)))
        self.rng = jax.random.key(seed)
        self.plain = jax.jit(lambda key,obs: policy.action(key,obs,5))
        self.guided = jax.jit(lambda key,obs,previous,d,end:
            policy.realtime_action(key,obs,5,previous,d,end,"exp",5.))

    def next_key(self):
        self.rng,key = self.jax.random.split(self.rng)
        return key

    def describe(self):
        task = Path(self.level_name).stem
        return dict(protocol_version=PROTOCOL_VERSION, control_modes=CONTROL_MODES.copy(),
                    task=task,available_tasks=[task],horizon=self.horizon, action_dim=self.action_dim,
                    observation_dim=self.observation_dim, level=self.level_name,
                    level_sha256=self.level_sha256, checkpoint_sha256=self.checkpoint_sha256,
                    parameters=self.parameter_count, jax_version=self.jax.__version__,
                    devices=[str(d) for d in self.jax.devices()])

    def _observation(self, request):
        if "task" in request and request["task"] != Path(self.level_name).stem:
            raise ValueError("Request task does not match policy level")
        obs = self.np.asarray(request["observation"], dtype=self.np.float32)
        if obs.shape != (self.observation_dim,) or not self.np.isfinite(obs).all():
            raise ValueError("Invalid observation shape or values")
        return self.jnp.asarray(obs)

    def initialize(self, request):
        obs = self._observation(request)
        seed = request["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Invalid episode seed")
        key = self.jax.random.key(seed)
        calibration_ms = self.calibrate(obs)
        self.plain(key,obs[None]).block_until_ready()
        self.rng = key
        chunk = self.plain(self.next_key(),obs[None])[0]
        chunk.block_until_ready()
        return dict(actions=self.np.asarray(chunk).tolist(), observation_tick=0,
                    calibration_inference_ms=calibration_ms,task=Path(self.level_name).stem)

    def calibrate(self, obs):
        """A causal warmed guided-inference measurement, before episode execution."""
        key = self.jax.random.key(0)
        previous = self.jnp.zeros((1,self.horizon,self.action_dim))
        self.guided(key,obs[None],previous,0,self.horizon).block_until_ready()
        start = time.perf_counter()
        self.guided(key,obs[None],previous,0,self.horizon).block_until_ready()
        return (time.perf_counter()-start)*1000

    def infer(self, request):
        jnp, np = self.jnp, self.np
        obs = self._observation(request)
        previous = np.asarray(request["previous_actions"], dtype=np.float32)
        d, end = request["predicted_steps"], request["overlap_steps"]
        tick = request["observation_tick"]
        method = request.get("method", "rtc")
        if (previous.shape != (self.horizon,self.action_dim) or not np.isfinite(previous).all()
                or type(d) is not int or type(end) is not int or type(tick) is not int or tick < 0):
            raise ValueError("Invalid action shape, values, or scheduling fields")
        if not 0 <= d <= end <= self.horizon or method not in CONTROL_MODES:
            raise ValueError("Invalid method or prefix/overlap bounds")
        start = time.perf_counter()
        if method in {"rtc","wireless_rtc"}:
            chunk = self.guided(self.next_key(),obs[None],jnp.asarray(previous)[None],d,end)[0]
        else:
            chunk = self.plain(self.next_key(),obs[None])[0]
        chunk.block_until_ready()
        return dict(actions=np.asarray(chunk).tolist(), observation_tick=tick,
                    inference_ms=(time.perf_counter()-start)*1000,task=Path(self.level_name).stem)


def run_experiment(config, checkpoint, output, metadata, viewer=None, view_pace=False):
    if config["episodes"] < 1 or config["min_execution_steps"] < 1:
        raise ValueError("episodes and min_execution_steps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    backend = KinetixPolicy(checkpoint,config["level"],config["seed"])
    if viewer is not None:
        viewer.set_renderer(backend.make_frame_renderer())
    jax,jnp,np = backend.jax,backend.jnp,backend.np
    H = backend.horizon
    if config["min_execution_steps"] >= H:
        raise ValueError("min_execution_steps must be smaller than H")
    metadata.update(kind="pretrained_kinetix_logical_time", checkpoint=str(Path(checkpoint).resolve()),
                    checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    parameters=backend.parameter_count, jax_version=jax.__version__,
                    devices=[str(d) for d in jax.devices()],
                    fallback="zero action with physics still advancing",
                    delay_units="ms converted with ceil to controller steps",
                    config=config)
    metadata["viewer"] = dict(enabled=viewer is not None, paced=view_pace,
                              fps=viewer.fps if viewer is not None else None)
    metadata["control_modes"] = CONTROL_MODES.copy()
    (output/"manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    summaries = []
    for method in config.get("methods",list(CONTROL_MODES)):
        if method not in CONTROL_MODES:
            raise ValueError("Unknown control method")
        for predictor_name,predictor_config in predictors_for(method,config):
            # Reset policy/environment randomness for each method/predictor.
            backend.rng = jax.random.key(config["seed"])
            request_rows, episodes = [], []
            for episode in range(config["episodes"]):
                predictor = make_predictor(predictor_config) if predictor_config is not None else None
                backend.rng = jax.random.fold_in(jax.random.key(config["seed"]),episode)
                # Same exogenous trace per episode across methods. No current-target leakage.
                delays = generate_delays(config["delay"], int(backend.params.max_timesteps)+1,
                                         config["seed"]+episode)
                env_key = jax.random.fold_in(jax.random.key(config["seed"]),episode)
                env_key,reset_key = jax.random.split(env_key)
                obs,state = backend.reset(reset_key)
                calibration_ms = backend.calibrate(obs)
                if method == "rtc":
                    predictor.update(calibration_ms)
                previous = np.asarray(backend.plain(backend.next_key(),obs[None])[0])
                valid, tick, solved, exhausted_total = H, 0, False, 0
                if viewer is not None:
                    viewer.update(state, phase="Running", method=method, predictor=predictor_name,
                                  episode=episode+1, episodes=config["episodes"], tick=0,
                                  simulation_ms=0, success=False,predictor_basis=CONTROL_MODES[method],
                                  predicted_ms=None,inference_ms=None)
                for request_index,actual_ms in enumerate(delays):
                    prediction_ms = predictor.predict() if predictor is not None else None
                    raw_d = delay_steps(prediction_ms,config["control_ms"]) if prediction_ms is not None else 0
                    d = min(raw_d,valid)  # log unsupported predicted prefix, do not hide it
                    key = backend.next_key()
                    start = time.perf_counter()
                    if method in {"rtc","wireless_rtc"}:
                        new = backend.guided(key,obs[None],jnp.asarray(previous)[None],d,valid)[0]
                    else:
                        new = backend.plain(key,obs[None])[0]
                    new.block_until_ready()
                    inference_ms = (time.perf_counter()-start)*1000
                    new = np.asarray(new)
                    actual_d = delay_steps(actual_ms,config["control_ms"])
                    # Single in-flight request: next start is no earlier than response arrival.
                    elapsed = max(config["min_execution_steps"],actual_d)
                    steps = min(elapsed,int(backend.params.max_timesteps)-tick)
                    actions,exhausted = execute_interval(previous,new,actual_d,steps,valid)
                    request_tick = tick
                    done = False
                    executed = 0
                    for action in actions:
                        step_started = time.monotonic() if view_pace else None
                        env_key,step_key = jax.random.split(env_key)
                        obs,state,reward,done,info = backend.step(step_key,state,jnp.asarray(action))
                        tick += 1
                        executed += 1
                        solved = solved or bool(info.get("success",reward > 0))
                        if viewer is not None:
                            viewer.update(state, phase="Running", tick=tick,
                                          simulation_ms=tick*config["control_ms"], success=solved,
                                          predicted_ms=prediction_ms, actual_ms=actual_ms,
                                          inference_ms=inference_ms,predictor_basis=CONTROL_MODES[method])
                        if view_pace:
                            time.sleep(max(0,config["control_ms"]/1000 - (time.monotonic()-step_started)))
                        if bool(done):
                            break
                    exhausted = sum(1 for k in range(executed) if k < actual_d and k >= valid)
                    exhausted += sum(1 for k in range(executed) if k >= actual_d and k >= H)
                    exhausted_total += exhausted
                    request_rows.append(dict(episode=episode,request=request_index,tick=request_tick,
                        actual_ms=actual_ms,predicted_ms=prediction_ms,actual_steps=actual_d,
                        predicted_steps=raw_d,applied_prefix_steps=d,overlap_steps=valid,
                        prefix_clipped=raw_d>valid,execution_steps=elapsed,
                        executed_steps=executed,buffer_exhaustion_steps=exhausted,
                        predictor_basis=CONTROL_MODES[method],
                        predictor_observation_ms=predictor_sample(method,inference_ms,actual_ms),
                        prediction_exceeded=actual_d>raw_d if predictor is not None else None,stale_response=actual_d>=H,
                        host_inference_ms=inference_ms))
                    if predictor is not None:
                        predictor.update(predictor_sample(method,inference_ms,actual_ms))
                    valid = max(0,H-elapsed)
                    previous = np.concatenate([new[elapsed:],np.zeros((min(elapsed,H),backend.action_dim))])
                    if bool(done) or tick >= int(backend.params.max_timesteps):
                        break
                episodes.append(dict(episode=episode,success=solved,steps=tick,
                                     buffer_exhaustion_steps=exhausted_total,
                                     calibration_inference_ms=calibration_ms if method == "rtc" else None))
            directory = output/condition_name(method,predictor_name)
            directory.mkdir()
            for filename,rows in [("requests.csv",request_rows),("episodes.csv",episodes)]:
                with (directory/filename).open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            summary=dict(method=method,predictor=predictor_name,episodes=len(episodes),
                         solve_rate=sum(r["success"] for r in episodes)/len(episodes),
                         requests=len(request_rows),
                         predictor_basis=CONTROL_MODES[method],
                         prediction_exceedance_rate=sum(r["prediction_exceeded"] for r in request_rows)/len(request_rows) if method != "naive" else None,
                         clipped_prefix_rate=sum(r["prefix_clipped"] for r in request_rows)/len(request_rows) if method != "naive" else None,
                         buffer_exhaustion_steps=sum(r["buffer_exhaustion_steps"] for r in episodes))
            summaries.append(summary)
            print(json.dumps(summary),flush=True)
            (directory/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    if viewer is not None:
        viewer.update(phase="Completed. Last frame retained; Ctrl+C in the server terminal closes the viewer.")
