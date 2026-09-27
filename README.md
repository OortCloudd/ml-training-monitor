# ML Training Monitor

Give your coding agent the tools to monitor, profile and optimize your training
on your own infrastructure. The agent connects your existing logs, adapts the
collectors and profiling workflow to your machines and GPUs, investigates
bottlenecks, and checks the effect of its changes.

The package includes the dashboard and collectors: progress, loss, gradients,
device activity, input preparation, checkpoint/evaluation costs, and optional
CPU/GPU profiles. Training-specific paths and hooks are supplied through
configuration and an explicit training-loop interface. Optional decision support
lets an agent prepare concrete interventions for the user's review.

[SKILL.md](SKILL.md) is the unified agent entry for this dashboard, runtime
inspection, MFU and performance engineering. All existing collection and
optional review workflows remain available. See [unification](docs/unification.md)
for the integration map and migration procedure.

## Give it to your agent

Point your agent at [SKILL.md](SKILL.md) in this checkout. For example:

> Use this skill to connect the dashboard to my training environment. Discover
> my machines, available GPUs, launchers and existing logs, implement the
> necessary adapters, and verify the displayed measurements. Then investigate
> bottlenecks and validate optimizations within the scope we agree on.

[Environment adaptation](docs/environment-adaptation.md) identifies the reusable
interfaces and the work needed for local, multi-GPU and cluster deployments.
The agent performs that integration; a list of visible GPUs alone does not
establish working distributed monitoring. The optimization method includes
profiling again after a change to identify the remaining cost.

No demonstration workload is required. Connect real training logs directly.
Any synthetic example or test fixture is illustrative and is not evidence of
training speed or a successful deployment on another cluster.

## What you can use

| Capability | What it needs |
| --- | --- |
| Training curves, progress, ETA and checkpoint state | Existing logs, or `Monitor.step()` |
| Useful compute throughput / MFU, with estimates or bounds | Explicit FLOP/timing/hardware contract; optional per-update counts |
| GPU activity/power/clocks/VRAM tables and raw rolling samples | NVIDIA `nvidia-smi`; independent of training hooks |
| CPU/RAM and selected filesystem capacity | Linux; explicitly configured filesystems |
| Input, forward/backward and optimizer host timing | `Monitor.phase()` around the phases you want to observe |
| Checkpoint, validation and other work outside updates | `Monitor.operation()` |
| Diagnostic history, a selectable observation, and model details | Your logged diagnostic scalars and model metadata |
| Detailed CPU/GPU interval breakdown and capture history | Optional scheduled PyTorch capture |
| Nsight kernel counters and measured rooflines | Existing reports, or the optional checkpoint replay worker |
| Measured investigation leads and reviewed agent proposals | Explicit opt-in to decision support; user approval for each intervention |

Ordinary monitoring requires no training pause, checkpoint reload, CUDA
synchronization, PyTorch import by the server, or administrator access. The
light phase timers measure host wall time. They are useful without a profiler,
but may overlap GPU execution and must not be added as exclusive percentages.

## Start with your existing logs

Python 3.10+; the dashboard itself uses the standard library. From this folder:

```bash
git clone https://github.com/OortCloudd/ml-training-monitor.git
cd ml-training-monitor
cp monitor.example.json monitor.local.json
python3 -m mlmonitor --config monitor.local.json --check
python3 -m mlmonitor --config monitor.local.json
```

Open `http://127.0.0.1:8790`. Add runs to the local configuration:

```json
{
  "id": "my-run",
  "name": "My training run",
  "directory": "/path/to/my/run",
  "files": {"telemetry": "metrics.jsonl"},
  "fields": {
    "step": "global_step",
    "loss": "train.loss",
    "seconds": "timing.update_seconds",
    "gradient": null
  }
}
```

Only the step counter is required for progress. Missing metrics remain missing.
The browser starts with your actual hardware and an empty run list; there is
no demo data. Relative paths resolve against the config file, then each run's
directory. See [integration.md](docs/integration.md) for all field contracts.

## Add phase monitoring to a training loop

Make this local package available in the training environment, for example with
`pip install -e /path/to/ml-training-monitor`. Your existing loop keeps control
of data loading, optimization, checkpointing and validation.

```python
from mlmonitor import Monitor

monitor = Monitor(
    "/path/to/monitor-output",
    run_id="my-run",
    config=training_config,          # hashed locally; raw config is not exported
    target_updates=total_updates,
    start_step=completed_updates,    # your authoritative resume counter
    details={"Model": model_name, "Precision": precision},
)

for update in range(completed_updates + 1, total_updates + 1):
    with monitor.step(update) as observation:
        with monitor.phase("input_wait"):
            batch = next(batches)
        with monitor.phase("forward_host"):
            loss = compute_loss(model, batch)
        with monitor.phase("backward_host"):
            loss.backward()
        with monitor.phase("optimizer"):
            optimizer.step()
            optimizer.zero_grad()
        observation.record(loss=already_logged_loss)

    if your_existing_checkpoint_condition:
        with monitor.operation("checkpoint"):
            checkpoint_path = save_your_checkpoint()
        monitor.checkpoint(update, checkpoint_path)

monitor.close()
```

This is an integration pattern, not a replacement training loop. Keep your
accumulation, scaler, clipping, optimizer order, logging frequency, and resume
semantics. Pass scalars already available to your logger; do not introduce a
GPU `.item()` solely to populate the dashboard. The default step duration is
host wall time around that scope; `step_seconds=` can supply your existing
timing measurement with its established boundaries.

The generated files connect directly to the dashboard: register the run ID and
monitor output directory, with no field mappings needed. Phase summaries are
published every five seconds by default, with a rolling 100-update window;
`flush()` publishes immediately. Lightweight logging still has some I/O cost;
measure it on your workload before claiming negligible overhead.

## Profiling is a separate choice

`profile_every=30000` opts into a bounded PyTorch capture at those optimizer
updates. `profile_steps=[...]` selects particular updates. These captures
synchronize the selected device and perturb timing; ordinary steps do not.
Captures remain dated and linked to their training configuration.

`heavy_every=30000` only creates a request after your code reports a durable
checkpoint at a due update. It does not interrupt training. A separately
invoked worker handles Nsight using an adapter for your own checkpoint/GPU
ownership/recovery workflow. See [profiling.md](docs/profiling.md).

Both are disabled by default. Leaving them disabled preserves all normal
monitoring, host phase timing, diagnostics, and outside-step timing. There is
no global 30k policy and no imported assumption about your scheduler.

## Optional decision support

Enable this only when the user wants it. The built-in engine suggests a few
investigation leads from recorded costs. An agent can turn a lead—or its own
code analysis—into a concrete proposal with the change, evidence, expected
effect, validation, risks and rollback. The user approves or rejects that exact
scope in the dashboard. Approval does not execute a training command.

```json
"decision_support": {"enabled": true, "directory": "local/proposals"}
```

The [decision-support guide](docs/decision-support.md) explains submission,
review, revocation and agent-reported outcomes. Ordinary monitoring works with
this mode disabled. The suggestions are transparent heuristics, not a promise
of a speedup or an automatic causal diagnosis. No LLM service is required.

## For coding agents

[AGENTS.md](AGENTS.md) maps each integration point to its configuration or hook.
The [agent access guide](docs/agent-access.md) explains how to read the same
measurements as the dashboard and gives a short optimization method. The agent
decides how to investigate and improve the user's training; the monitor supplies
the data and instrumentation.

Real-time metric quality is explicit: source/units, observation window, valid
sample counts, collection gaps, freshness and timestamp provenance. Dated
profiles stay separate from live readings. Optional proposal review leaves the
user in control of what is authorized; agents remain responsible for examining
the actual code and reporting measured results.

## Verification and limits

```bash
python3 -m unittest discover -s tests -v
```

Tests cover collection, interval accounting, diagnostic histories, optional
capture scheduling, durable-checkpoint requests, and worker cleanup/deduplication.
Fixtures are synthetic tests, not performance measurements. The supplied
collectors observe a single host, including multiple local GPUs. Agents adapt
multi-host collection and distributed per-rank profiling to the user's actual
environment; see [environment adaptation](docs/environment-adaptation.md) for
implementation boundaries and verification. NVIDIA is the GPU collector
implemented here. The web server has
no authentication: keep loopback or use your established authenticated tunnel.

Apache-2.0. See [LICENSE](LICENSE). This project is separate from the
[historical ML Performance Engineering repository](https://github.com/OortCloudd/ml-performance-engineering);
its investigation guidance and trace helper are now included here, under the
unified [ML Training Monitor skill](SKILL.md). No second skill installation is
needed. Existing skill-name callers can use a compatibility redirect during
migration; preserve the older repository as provenance rather than maintaining
two independent methods.

To make an installed checkout discoverable as a Codex skill, link its directory
into `~/.codex/skills/ml-training-monitor`. Keep that installation at a validated
revision; prepare updates in a separate development checkout or Git worktree.
The Python package and skill remain one maintained codebase. This separates
deployment from development without maintaining a second dashboard or a private
fork of the method. See [deployment updates](docs/unification.md#deployment-and-development).
