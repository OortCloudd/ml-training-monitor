# Optional profiling, with the collection workflow included

Light monitoring is the default. There are two independent opt-ins.

## Framework captures within training

Set `profile_every=N` or `profile_steps=[...]` on `Monitor`. A due update is
profiled once; an existing per-update capture prevents an automatic duplicate.
Resuming at a completed milestone does not replay that milestone or trigger an
immediate capture. Cadence is expressed in optimizer updates, not microbatches.

The supplied `TorchCapture` collects CPU/CUDA events on one selected device.
It synchronizes that device at the capture boundaries, so captured timings are
perturbed. No extra synchronization occurs in light mode. Multi-device work
requires one explicitly selected device per capture; its result is not a sum
of concurrent GPU times. PyTorch and its profiler are imported only when a
capture is due.

`phase()` annotations and actual device events populate the existing detailed
phase table. Device execution takes precedence over overlapping host ranges;
uncovered time remains other/unclassified host intervals. GPU gaps do not prove
CPU computation, and activity does not establish achieved FLOPs or bandwidth.
The latest capture, dated history, configuration identity and recent unprofiled
baseline are saved. Comparing with the recent baseline is a perturbation
indicator, not a controlled speedup estimate.

PyTorch's Kineto event access varies by version. The current implementation
uses `profile.profiler.kineto_results.events()`; failure disables detailed
captures for that instance and is reported. This path needs qualification in
the user's installed framework/GPU environment before a long run.

## Checkpoint replay and Nsight

Set `heavy_every=N` or `heavy_steps=[...]`. After an existing durable checkpoint
save, call `monitor.checkpoint(update, checkpoint_path, terminal=...)`. A small
deduplicated request is created for that run/config/update. This call does not
launch Nsight, pause training, or load weights.

Run the separate worker only when that profiling workflow is intended:

```bash
python3 -m mlmonitor.worker \
  --run-directory /path/to/monitor-output \
  --adapter /path/to/your_project/monitor_adapter.py \
  --timeout 120
```

One invocation checks pending requests once. It can be invoked by the user's
existing scheduler; the dashboard does not install one. The worker already
provides:

- checkpoint path/size/mtime consistency checks and one receipt per attempted capture;
- a per-run lock preventing concurrent workers;
- minimum free-space, duration and raw-output bounds;
- bounded Nsight collection of selected kernel launches;
- exact-name/unit CSV import into the dashboard's counter format;
- success/failure receipts and cleanup through the adapter context;
- an explicit recovery entry point for interrupted workers.

## The adapter supplies your workflow, not a second profiler

An adapter is trusted local Python code selected explicitly on the worker CLI.
It is never loaded by the web server. Implement these two functions:

```python
from contextlib import contextmanager
from mlmonitor.worker import CapturePlan

def ready(request):
    # Read-only check: durable checkpoint available, expected job/GPU ownership,
    # and a bounded capture can proceed under your scheduling rules.
    return your_workflow.is_ready_for_capture(request)

@contextmanager
def capture_context(request):
    token = None
    try:
        # The workflow atomically verifies ownership and obtains the GPU.
        # If needed, it pauses at its own safe boundary and records recovery.
        token = your_workflow.acquire_for_capture(request)
        yield CapturePlan(
            command=your_workflow.replay_command(request),
            gpu_uuid=your_workflow.gpu_uuid(token),
            kernel_regex=your_workflow.selected_kernel_pattern(request),
            scope="Your actual replay shape/precision/warmup and kernel scope",
            launch_count=4,
        )
    finally:
        if token is not None:
            your_workflow.release_and_resume(token)
```

`your_workflow` represents the methods your project already owns; it is not a
package to install. The small adapter is where an agent connects your scheduler,
checkpoint format, replay arguments, and GPU ownership. The worker constructs
the Nsight command and imports the output; it does not ask the agent to rebuild
the collection pipeline.

`ready` must not pause a job. The context must recheck ownership on acquisition;
a read-only readiness check alone cannot prevent a scheduler race. A terminal
request must wait for a free GPU, reuse the appropriate final input/schedule,
and never preempt a successor. Preserve the production checkpoint, optimizer,
source cursor and RNG. Never promote replay weights. The adapter chooses a
representative warmed replay and kernel pattern from actual timeline evidence.

The worker fixes `CUDA_VISIBLE_DEVICES` to the leased GPU UUID and exports a
bounded set of counters. It does not acquire administrator rights or change
driver settings. Counter access remains an environment requirement. Review raw
reports locally; they may contain command/path metadata.

Use a retained checkpoint path that will not be overwritten while the request
is pending. Size/mtime checks do not validate a model format or prove the
checkpoint's embedded run identity; the replay adapter must do that using the
project's actual checkpoint contract.

On timeout, collection/import error or SIGTERM, the context exits so its
`finally` runs. SIGKILL or a machine failure cannot run Python cleanup. Keep
your recovery journal in the scheduler/adapter and implement
`recover(request, receipt)`. Invoke `--recover` explicitly to handle receipts
still marked RUNNING; it does not automatically retry captures. Failed
attempts also receive receipts and are not silently repeated.

## Existing reports and rooflines

You can import a report without using the worker:

```bash
LC_ALL=C ncu --import your-report.ncu-rep --page raw --csv > your-report.csv
python3 -m mlmonitor.import_ncu your-report.csv \
  --manifest your-manifest.json --output /path/to/monitor-output/nsight/capture.json
```

The manifest requires the real `run_id`, ISO-8601 `captured_at` with timezone,
`gpu_uuid`, `config_sha256`, `tool_version`, and `scope`. Optional
`checkpoint_update` and `profile_overhead_pct` must be actual measured metadata.
Unsupported counters remain absent. Reports are interpreted per kernel, not as
whole-training averages. Nsight replay/cache/clock effects must be stated when
interpreting timing.

Rooflines require compatible measured FLOPs and memory traffic, arithmetic
intensity, precision, operation counting, and memory-level ceilings. The
importer does not infer those from activity percentages. Supply the optional
validated kernel `roofline` object only when those measurements exist.

Primary metric references: [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
[CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html),
and [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
