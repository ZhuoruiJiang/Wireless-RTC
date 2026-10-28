# Wireless RTC

Evaluate causal response-delay prediction for Real-Time Chunking (RTC).
The laptop owns observations, delay prediction, and execution; a server can serve the
pretrained policy. This first milestone provides a runnable prediction study,
network probes, and **offline pretrained Kinetix RTC with fixed/random delays**.

**Status:** live probes, a policy RPC endpoint, and a paced simulator-to-server
controller are implemented. `rtc` runs offline with injected logical-time delays;
`rtc-live` runs local physics with remote policy inference and measured RPC delays.
Synthetic prediction results must not be described as campus WiFi or robot success results.

## Structure

| Folder | Responsibility |
|---|---|
| `src/wireless_rtc/network/` | Delay sources, persistent TCP client, network probes |
| `src/wireless_rtc/protocol/` | Bounded JSON request/response messages |
| `src/wireless_rtc/server/` | Probe service and optional policy endpoint |
| `src/wireless_rtc/client/` | Delay conversion, action alignment, paced asynchronous live controller |
| `src/wireless_rtc/prediction/` | Recent maximum, rolling quantile, last-value predictors |
| `src/wireless_rtc/simulation/` | Pretrained upstream RTC/Kinetix adapter |
| `src/wireless_rtc/analysis/` | Causal scoring, CSV/JSON summaries, SVG plots |
| `configs/` | First-run experiment settings |
| `tests/` | Leakage, timing, protocol, optional pretrained checks |
| `scripts/` | Download one checkpoint |
| `vendor/rtc-kinetix/` | Unmodified pinned upstream submodule |
| `results/` | Small checked-in example results; use `runs/` for new runs |

## Quick start: prediction pipeline (Windows or Linux, Python 3.11)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first,
and Git. For a new checkout:

```sh
git clone https://github.com/ZhuoruiJiang/Wireless-RTC.git
cd Wireless-RTC
```

Run the following from the repository root. Deactivate any unrelated virtual
environment before starting. Each collaborator creates their own `.venv`;
do not copy or share a virtual-environment directory between machines.

```sh
uv venv --python 3.11 .venv
uv pip install -e .
uv run --no-sync python -m wireless_rtc prediction-suite --config configs/first_run.json --output runs/prediction-001
uv run --no-sync python -m unittest discover -s tests -v
```

No third-party runtime dependencies are needed for prediction/probing.
This creates a project-local `.venv` using Python 3.11, which matches the pinned
RTC dependency stack. `uv pip install` targets that environment, and the editable
installation picks up source changes without reinstalling. `.venv` is gitignored.
The project configures uv to copy installed files instead of hardlinking from its
cache, which avoids OneDrive's incompatible-hardlink errors. No activation or
`.deps/client` environment is needed for the documented setup.

No shell activation is required. Use `uv run --no-sync` throughout this README
to execute commands in `.venv` while preserving the optional RTC packages
installed below. Running `uv sync` or plain `uv run` can remove those packages
because they are not declared in the minimal `pyproject.toml`.

Check the selected interpreter with:

```sh
uv run --no-sync python -c "import sys; print(sys.executable)"
```

Install additional packages with `uv pip install <package>`. These instructions
use uv's virtual-environment and pip interface; RTC dependency versions are
listed in `requirements/rtc-cpu.txt`, rather than a project `uv.lock`.

Open `comparison.csv` and each `prediction.svg` under the output folder.
Each source has 1,000 requests; 32 are warmup. Configurations are fixed in advance.
Use a new output directory each run.

## Set up a laptop for live simulation

Complete the quick-start `.venv` setup above. These additional commands work from
the repository root on Windows or Linux and install the CPU physics/rendering
stack into that **same `.venv`**:

```sh
git submodule update --init --recursive
uv pip install --link-mode copy -e . -r requirements/rtc-cpu.txt
uv pip install --link-mode copy --no-deps -e vendor/rtc-kinetix/third_party/kinetix "jaxued @ git+https://github.com/DramaCow/jaxued.git@62d155ae772aa2a0ecac9541d498df7ae88e6def"
```

The upstream packages deliberately use `--no-deps`: their original dependency
declarations conflict with the pinned working stack. Install the CPU requirements
first, then these two upstream packages. The laptop needs no policy checkpoint,
CUDA toolkit, or administrator privileges. Do not run this CPU setup over a
working CUDA-enabled **server** environment.

Verify the interpreter, command, dependencies, and actual simulator initialization:

```sh
uv run --no-sync python -c "import sys; print(sys.executable)"
uv run --no-sync python -m wireless_rtc rtc-live --help
uv run --no-sync python -c "from wireless_rtc.simulation.kinetix import KinetixSimulator; s=KinetixSimulator(); print('Simulator ready:', s.observation_dim, 'observation values,', s.action_dim, 'actions')"
uv run --no-sync python -m unittest discover -s tests -v
```

The interpreter must be inside this checkout's `.venv` (`Scripts/python.exe` on
Windows; `bin/python` on Linux). Checkpoint tests skip unless `RTC_CHECKPOINT` is
set; the client setup does not require it. Follow
[the live pipeline instructions](#live-client-simulation-with-server-policy-inference)
below to start the server, SSH tunnel, and local viewer. If viewer port 8080 is
occupied, pass `--view-port 18080` and open <http://localhost:18080>.

### Existing Windows environment fails with access denied

If installing into an old `.venv` fails to remove package metadata, stop processes
using that environment and preserve it before rebuilding. From the project root
in PowerShell, choose an unused backup name:

```powershell
New-Item -ItemType Directory -Force .deps | Out-Null
Move-Item -LiteralPath .venv -Destination .deps/venv-backup
uv venv --python 3.11 .venv
```

Then repeat the laptop installation and verification commands above. The `.deps`
folder is ignored by Git. This preserves the old environment without deleting it
or changing its access permissions. Continue using `uv run --no-sync` with the new
standard `.venv`.

## Measure Cybertron latency

On Cybertron, repeat the virtual-environment setup above in its repository checkout, then run the probe server:

```sh
uv run --no-sync python -m wireless_rtc serve --host 127.0.0.1 --port 8765
```

On your laptop, create an SSH tunnel in a separate terminal:

```sh
ssh -N -L 8765:127.0.0.1:8765 zhuorui@cybertron5.csres.utexas.edu
```

Then measure from your laptop:

```sh
uv run --no-sync python -m wireless_rtc probe --host 127.0.0.1 --port 8765 --samples 300 --output runs/campus-001
uv run --no-sync python -m wireless_rtc evaluate-trace --trace runs/campus-001/trace.csv --output runs/campus-eval-001
```

RTT includes serialization, transport, server processing, and decoding, but excludes
initial connection establishment. SSH adds transport overhead; report that consistently.
Use the same persistent connection for successive requests. Default probes echo 1 KiB;
match eventual observation/action sizes before drawing deployment conclusions.
The server binds to loopback by default; the protocol has no standalone authentication.
The service supports one experimental client at a time.

For a local transport check, run server and probe on the same computer.
`probe --delay fixed` adds 40 ms of artificial server waiting;
`--delay random` adds seeded uniform 10-70 ms server waiting.
These are not measurements of wireless jitter. Errors/timeouts are logged, not dropped;
offline trace evaluation deliberately rejects censored rows until handled explicitly.

## Pretrained RTC / Kinetix

Complete the virtual-environment setup above first. Install the CPU simulation
dependencies into the same `.venv` using the commands below. These experiments
run locally without a remote server; installation and checkpoint download need
Internet access.

Keep upstream unmodified. Pinned RTC revision:
`9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`.

```sh
git submodule update --init --recursive
uv pip install -r requirements/rtc-cpu.txt
uv pip install --no-deps -e vendor/rtc-kinetix/third_party/kinetix
uv pip install --no-deps "jaxued @ git+https://github.com/DramaCow/jaxued.git@62d155ae772aa2a0ecac9541d498df7ae88e6def"
uv run --no-sync python scripts/download_checkpoint.py --epoch 24 --level grasp_easy
uv run --no-sync python -m wireless_rtc rtc --config configs/rtc_fixed.json --checkpoint checkpoints/24/policies/worlds_l_grasp_easy.pkl --output runs/rtc-fixed-001
uv run --no-sync python -m wireless_rtc rtc --config configs/rtc_random.json --checkpoint checkpoints/24/policies/worlds_l_grasp_easy.pkl --output runs/rtc-random-001
```

This runner uses the actual pretrained upstream flow policy and guided RTC;
it does not replace the policy with a toy controller.
Predicted delay and injected actual delay are independent.
Output includes per-request timings/alignment, per-episode success, checkpoint hash,
parameter count, configuration, software revision, and JAX device.
The first inference includes JIT compilation; `host_inference_ms` is diagnostic,
not a warmed-up inference benchmark.
Five episodes per condition are only a functional smoke test.

### Watch a run in your browser

The optional viewer renders the actual Kinetix physics state without a desktop,
X server, or browser on Cybertron. Run from the repository root on the server:

```sh
uv run --no-sync python -m wireless_rtc rtc --config configs/rtc_fixed.json --checkpoint checkpoints/24/policies/worlds_l_grasp_easy.pkl --output runs/rtc-view-001 --view --view-port 8080 --view-pace
```

If you require GPU execution, prefix that command with `JAX_PLATFORMS=cuda`
after verifying your CUDA-enabled JAX environment. Rendering uses Pillow, already
included by the RTC dependency stack; if missing, use `uv pip install Pillow`.
Do not reinstall the CPU requirements over a working GPU environment.

In a separate terminal **on your laptop**, keep this SSH tunnel running:

```sh
ssh -N -o ExitOnForwardFailure=yes -L 8080:127.0.0.1:8080 zhuorui@cybertron5.csres.utexas.edu
```

Open <http://localhost:8080> on your laptop. The page shows simulation frames plus
method, predictor, episode, simulation tick, success, and injected/predicted delay.
The viewer starts before loading/compiling the policy; the first frame can take
a while, especially during JAX compilation. Render errors appear on the page.
After completion, the viewer retains the final frame and stays open until Ctrl+C.
The canvas viewer uses a single `/api/snapshot` endpoint that delivers simulation
progress and new frames together. Images retain the tick/episode metadata of the
state they depict. Each browser tracks a session ID and frame number, so it can
join after completion or reconnect after the server restarts. The former image
and MJPEG endpoints have been replaced.

The page explicitly shows connecting, warming, live, completed, disconnected, and
render-error states. **Pause view** freezes only the display; physics continues.
**Reconnect now** retries immediately. The canvas retains the last good frame
through interruptions, and the frame caption identifies its tick. New HTML, CSS,
and JavaScript assets ship with the Python package; no frontend build or external
web service is required. Port conflicts produce an actionable command-line error.
If the experiment fails, the viewer remains open to show the error until Ctrl+C.
Open the page as soon as the viewer URL is printed to watch the run in progress.
**Completed** means the experiment has finished: its final image is intentionally
stationary. Only one viewer may own a port, including on Windows; stop an old
viewer with Ctrl+C or choose another `--view-port` for a new run.
Use a fresh output directory on each run. Ctrl+C during a run stops it; partial
output is not a completed experiment.

`--view-fps 10` caps rendering at ten frames per wall-clock second (range: >0 to
60). Rendering and browser delivery use separate threads and retain only the
latest state/frame, so slow viewers skip frames instead of queuing simulator
steps. `--view-pace` limits stepping to at most the configured control rate
(50 steps/s at 20 ms) so fast runs are watchable; compilation/inference can make
it slower. Omit pacing to run at full speed, which may skip entire short episodes.
This remains the **logical-time, injected-delay experiment**, not live policy RPC.

Viewing adds GPU/CPU contention and pacing overhead. Disable `--view` for timing
measurements. The manifest records whether viewing/pacing was enabled. The viewer
binds only to `127.0.0.1`; access it through SSH. It does not need a public port.

For server GPU use, install a compatible Linux CUDA JAX build for the server GPU/driver
instead of the CPU build. The upstream JAX 0.4.35 stack may need adaptation on newer GPUs;
do not assume it supports your laptop GPU. CPU is sufficient to begin functional tests.

Optional actual-checkpoint test (PowerShell):

```powershell
$env:RTC_CHECKPOINT = "checkpoints/24/policies/worlds_l_grasp_easy.pkl"
uv run --no-sync python -m unittest discover -s tests -v
```

To enable policy RPC on the server:

```sh
uv run --no-sync python -m wireless_rtc serve --checkpoint checkpoints/24/policies/worlds_l_grasp_easy.pkl
```

Requests carry observation, aligned previous actions, predicted prefix steps,
overlap steps, and observation tick. Responses echo tick/ID and inference timing.
Only load trusted pickle checkpoints. Checkpoints and large logs are gitignored.

## Live client simulation with server policy inference

The **laptop** owns Kinetix physics, delay prediction, action buffering, and the
browser viewer. The **server** owns the pretrained policy and checkpoint.
Only observations/action chunks cross the network; video frames stay on the laptop.
Use [the laptop setup](#set-up-a-laptop-for-live-simulation) above to prepare the
client's standard `.venv`; the client does **not** need a checkpoint. Use the same repository
revision and level on both machines. Preserve the CUDA-enabled environment on the
server; do not reinstall the CPU requirements there.

1. On **Cybertron**, from the updated repository, start the policy server:

   ```sh
   JAX_PLATFORMS=cuda uv run --no-sync python -m wireless_rtc serve --host 127.0.0.1 --port 8765 --checkpoint checkpoints/24/policies/worlds_l_grasp_easy.pkl
   ```

   Wait for `Listening on 127.0.0.1:8765`. CUDA must already be configured. The
   policy is warmed up when the client initializes its first episode.

2. On the **laptop**, keep this SSH tunnel open in its own terminal:

   ```sh
   ssh -N -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 zhuorui@cybertron5.csres.utexas.edu
   ```

3. In another **laptop PowerShell** terminal, run local CPU simulation and viewing:

   ```powershell
   $env:JAX_PLATFORMS = "cpu"
   uv run --no-sync python -m wireless_rtc rtc-live --config configs/rtc_remote.json --host 127.0.0.1 --port 8765 --output runs/rtc-live-001 --view
   ```

   Open <http://localhost:8080>. There is **no viewer-port SSH tunnel**. After the
   run completes, the local viewer keeps its final frame until Ctrl+C. Use a fresh
   output directory for each run. `--view-port` and `--view-fps` work as in `rtc`.
   Omit `--view` for timing measurements.

`rtc-live` is always paced; it has no `--view-pace` flag. The new remote config
starts with five episodes per condition, RTC/naive methods, and recent-max/p95
predictors. It deliberately has no `delay` section. Your offline fixed/random
configs retain their injected delays and are rejected by the live command.

Before each episode the client warms its physics code, the server warms inference
and resets the policy RNG, and an initial plain-policy action chunk is fetched.
This setup is outside the episode clock and predictor history. Thereafter, one
background worker sends policy requests while the main thread keeps executing
physics at the configured control interval. A response is considered only at a
controller boundary. The client drops the elapsed prefix, rejects expired chunks,
and executes zero actions when the buffer is exhausted. Physics is never paused
to wait for a running policy request. The final pending request is drained after
an episode ends so its result cannot leak into the next episode.

Each condition writes `requests.csv`, `ticks.csv`, `episodes.csv`, and `summary.json`.
The manifest includes the server checkpoint hash/devices, client devices, resolved
config, viewer settings, and run status. `requests.csv` records measured RTT
(`actual_ms`), predictions, server/inference timing, response IDs/ticks, actual
activation age (`elapsed_ticks`), discarded responses, and failures. RTT includes
worker dispatch, serialization, transport, inference, decoding, and validation;
it excludes initial connection establishment and episode warmup. The predictor
updates only after a successful response is observed during the episode, including
expired chunks, and resets each episode. Failed/censored delays never update it.

`ideal_delay_steps` is `ceil(RTT/control_ms)`. Actual chunk alignment uses simulation
ticks, which may differ when local physics/rendering is slow. `ticks.csv` records
buffer exhaustion, physics execution time, and controller-work budget overruns.
The controller does not run bursts of catch-up steps. Disable viewing and inspect
overruns before interpreting a run as meeting its nominal control rate. Live
conditions encounter different real network delays; they do not share the same
exogenous trace as offline conditions.

The default `--timeout 120` seconds allows first-time compilation. A timeout or
invalid response stops the run, preserves completed logs, and marks the manifest
as failed; it is not silently treated as a valid RTT or a successful episode.
Ctrl+C marks a run interrupted. Results from an incomplete run are not a completed
experiment. This controller is intended for the simulator, not robot hardware.

See [the experiment contract](docs/experiment.md) for timing, metrics, and later tuning.
Upstream reference: https://github.com/Physical-Intelligence/real-time-chunking-kinetix
