# Wireless RTC

Evaluate causal response-delay prediction for Real-Time Chunking (RTC).
The laptop owns observations, delay prediction, and execution; a server can serve the
pretrained policy. This first milestone provides a runnable prediction study,
network probes, and **offline pretrained Kinetix RTC with fixed/random delays**.

**Status:** live network probes and a policy RPC endpoint are implemented.
The Kinetix experiment runner currently uses logical-time delay injection on one
machine; a paced live simulator-to-server controller is the next milestone.
Synthetic prediction results must not be described as campus WiFi or robot success results.

## Structure

| Folder | Responsibility |
|---|---|
| `src/wireless_rtc/network/` | Delay sources, persistent TCP client, network probes |
| `src/wireless_rtc/protocol/` | Bounded JSON request/response messages |
| `src/wireless_rtc/server/` | Probe service and optional policy endpoint |
| `src/wireless_rtc/client/` | Delay conversion and action alignment |
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
then run these commands from the repository root. Deactivate any unrelated
virtual environment before starting:

```sh
uv venv --python 3.11
uv pip install -e .
uv run --no-sync python -m wireless_rtc prediction-suite --config configs/first_run.json --output runs/prediction-001
uv run --no-sync python -m unittest discover -s tests -v
```

No third-party runtime dependencies are needed for prediction/probing.
This creates a project-local `.venv` using Python 3.11, which matches the pinned
RTC dependency stack. `uv pip install` targets that environment, and the editable
installation picks up source changes without reinstalling. `.venv` is gitignored.

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

Open <http://localhost:8080> on your laptop. The page shows MJPEG frames plus
method, predictor, episode, simulation tick, success, and injected/predicted delay.
The viewer starts before loading/compiling the policy; the first frame can take
a while, especially during JAX compilation. Render errors appear on the page.
After completion, the viewer retains the final frame and stays open until Ctrl+C.
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

See [the experiment contract](docs/experiment.md) for timing, metrics, and later tuning.
Upstream reference: https://github.com/Physical-Intelligence/real-time-chunking-kinetix
