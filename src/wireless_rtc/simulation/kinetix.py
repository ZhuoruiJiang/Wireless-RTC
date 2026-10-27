"""Reuse upstream policies/physics; implement an explicit variable-delay scheduler.

This is an offline logical-time experiment. Host execution speed does not set
simulated delay. Live RPC integration is separate from this runner.
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


class KinetixPolicy:
    def __init__(self, checkpoint, level="worlds/l/grasp_easy.json", seed=0):
        self.jax, self.jnp, nnx, self.np, kenv, saving, model = dependencies()
        jax, jnp = self.jax, self.jnp
        self.level, static, self.params = saving.load_from_json_file(str(UPSTREAM / level))
        self.static = static.replace(screen_dim=(512,512))
        self.env = kenv.make_kinetix_env_from_name(
            "Kinetix-Symbolic-Continuous-v1", static_env_params=self.static)
        obs, _ = self.env.reset_to_level(jax.random.key(seed), self.level, self.params)
        self.action_dim = self.env.action_space(self.params).shape[0]
        self.policy = model.FlowPolicy(obs_dim=obs.shape[-1], action_dim=self.action_dim,
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
        self.reset = jax.jit(lambda key: self.env.reset_to_level(key,self.level,self.params))
        def step(key,state,action):
            key1,key2 = jax.random.split(key)
            noisy = action + jax.random.normal(key1,action.shape)*.1
            return self.env.step_env(key2,state,noisy,self.params)
        self.step = jax.jit(step)

    def next_key(self):
        self.rng,key = self.jax.random.split(self.rng)
        return key

    def make_frame_renderer(self):
        # Import/compile lazily on the viewer thread; normal runs pay no rendering cost.
        render = None

        def jpeg(state):
            nonlocal render
            import io
            from PIL import Image
            if render is None:
                from kinetix.render.renderer_pixels import make_render_pixels
                pixels = make_render_pixels(self.params, self.static.replace(downscale=2))
                # Match upstream video orientation: x/y transpose, then vertical flip.
                render = self.jax.jit(lambda s: pixels(s).round().clip(0,255)
                                      .astype(self.jnp.uint8).transpose(1,0,2)[::-1])
            frame = self.np.asarray(render(state))
            output = io.BytesIO()
            Image.fromarray(frame).save(output, format="JPEG", quality=80)
            return output.getvalue()

        return jpeg

    def infer(self, request):
        jnp, np = self.jnp, self.np
        obs = jnp.asarray(request["observation"], dtype=jnp.float32)
        previous = jnp.asarray(request["previous_actions"], dtype=jnp.float32)
        d, end = int(request["predicted_steps"]), int(request["overlap_steps"])
        if obs.ndim != 1 or previous.shape != (self.horizon,self.action_dim):
            raise ValueError("Invalid observation/action shape")
        if not 0 <= d <= end <= self.horizon:
            raise ValueError("Require 0 <= predicted_steps <= overlap_steps <= H")
        start = time.perf_counter()
        chunk = self.guided(self.next_key(),obs[None],previous[None],d,end)[0]
        chunk.block_until_ready()
        return dict(actions=np.asarray(chunk).tolist(),
                    observation_tick=request["observation_tick"],
                    inference_ms=(time.perf_counter()-start)*1000)


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
    (output/"manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    summaries = []
    for method in config.get("methods",["rtc","naive"]):
        if method not in {"rtc","naive"}:
            raise ValueError("Unknown control method")
        for predictor_name,predictor_config in config["predictors"].items():
            # Reset policy/environment randomness for each method/predictor.
            backend.rng = jax.random.key(config["seed"])
            request_rows, episodes = [], []
            for episode in range(config["episodes"]):
                predictor = make_predictor(predictor_config)
                backend.rng = jax.random.fold_in(jax.random.key(config["seed"]),episode)
                # Same exogenous trace per episode across methods. No current-target leakage.
                delays = generate_delays(config["delay"], int(backend.params.max_timesteps)+1,
                                         config["seed"]+episode)
                env_key = jax.random.fold_in(jax.random.key(config["seed"]),episode)
                env_key,reset_key = jax.random.split(env_key)
                obs,state = backend.reset(reset_key)
                previous = np.asarray(backend.plain(backend.next_key(),obs[None])[0])
                valid, tick, solved, exhausted_total = H, 0, False, 0
                if viewer is not None:
                    viewer.update(state, phase="Running", method=method, predictor=predictor_name,
                                  episode=episode+1, episodes=config["episodes"], tick=0,
                                  simulation_ms=0, success=False)
                for request_index,actual_ms in enumerate(delays):
                    prediction_ms = predictor.predict()
                    raw_d = delay_steps(prediction_ms,config["control_ms"])
                    d = min(raw_d,valid)  # log unsupported predicted prefix, do not hide it
                    key = backend.next_key()
                    start = time.perf_counter()
                    if method == "rtc":
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
                                          predicted_ms=prediction_ms, actual_ms=actual_ms)
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
                        prediction_exceeded=actual_d>raw_d,stale_response=actual_d>=H,
                        host_inference_ms=inference_ms))
                    predictor.update(actual_ms)
                    valid = max(0,H-elapsed)
                    previous = np.concatenate([new[elapsed:],np.zeros((min(elapsed,H),backend.action_dim))])
                    if bool(done) or tick >= int(backend.params.max_timesteps):
                        break
                episodes.append(dict(episode=episode,success=solved,steps=tick,
                                     buffer_exhaustion_steps=exhausted_total))
            directory = output/f"{method}-{predictor_name}"
            directory.mkdir()
            for filename,rows in [("requests.csv",request_rows),("episodes.csv",episodes)]:
                with (directory/filename).open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            summary=dict(method=method,predictor=predictor_name,episodes=len(episodes),
                         solve_rate=sum(r["success"] for r in episodes)/len(episodes),
                         requests=len(request_rows),
                         prediction_exceedance_rate=sum(r["prediction_exceeded"] for r in request_rows)/len(request_rows),
                         clipped_prefix_rate=sum(r["prefix_clipped"] for r in request_rows)/len(request_rows),
                         buffer_exhaustion_steps=sum(r["buffer_exhaustion_steps"] for r in episodes))
            summaries.append(summary)
            print(json.dumps(summary),flush=True)
            (directory/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    if viewer is not None:
        viewer.update(phase="Completed. Last frame retained; Ctrl+C in the server terminal closes the viewer.")
