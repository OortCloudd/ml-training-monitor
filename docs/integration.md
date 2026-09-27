# Training integration

## Dashboard configuration

`monitor.local.json` is deployment configuration, not training configuration.
The server reads it once at startup. Paths resolve relative to that file.

Top-level settings: `title`, `interval` (1–60 seconds, default 2),
`gpu_enabled` (default true), `gpu_indices` (null discovers all; list selects
physical `nvidia-smi` indices), `disks` (objects with `label` and `path`), and
`runs` (explicitly registered runs). Optional `decision_support` contains
`enabled` (default false) and `directory` for local proposal records. No automatic filesystem experiment scan.

A run needs `id` and `directory`. Optional `name`, `target_updates`, `step_unit`,
`details`, `config_sha256`, `capture_every`, and `capture_steps` describe it.
The latter two describe an external producer's schedule for display; they do
not start captures. The monitor's generated binding file supplies its own
schedule when those settings are omitted.

Default files within each run directory:

| `files` key | Filename | Producer |
| --- | --- | --- |
| `telemetry` | `training_telemetry.jsonl` | Existing logger or `Monitor.step()` |
| `state` | `state.json` | Existing producer or Monitor |
| `bindings` | `run_bindings.json` | Configuration identity and display metadata |
| `health` | `health_events.jsonl` | Existing diagnostics or `Monitor.diagnostic()` |
| `light_profile` | `phase_timings.json` | `Monitor.phase()` |
| `operations` | `operations.json` | `Monitor.operation()` |
| `profile` | `profile.json` | Latest scheduled framework capture |
| `profiles` | `profile_history.jsonl` | Framework capture history |
| `captures` | `nsight/` | Imported hardware summaries |

Override `files` entries to point at existing filenames or absolute paths. Set
optional entries to null when there is no producer. Point the dashboard at an
existing run directory to observe it; adding light hooks is a separate choice.

Default telemetry `fields`:

| Dashboard value | JSON field | Semantics |
| --- | --- | --- |
| `step` | `update` | Increasing completed-update counter |
| `loss` | `loss_components.loss` | Your objective, finite scalar |
| `seconds` | `update_seconds` | Seconds with the timing scope you establish |
| `gradient` | `encoder_gradient_norm` | Existing scalar norm, identify pre/post clipping |
| `timestamp` | `timestamp` | Unix seconds of observation; otherwise file mtime |
| `profiled` | `profiled` | Optional boolean indicating an instrumented update |
| `source_read_seconds` | same name | Optional source-read timer |
| `input_prepare_seconds` | same name | Optional producer preparation timer |
| `prepared_input_wait_seconds` | same name | Consumer wait for ready inputs |

Dot-separated mappings traverse nested objects. Optional fields can be null.
Arrays, booleans, non-finite numbers and missing scalars are not converted to
zero. JSONL records must end with a newline; partial records wait for the next
append. Rotation/truncation resets the corresponding history. Duplicate or
decreasing updates are not appended to the same curve. Use a separate run ID
and output directory for a new trajectory.

The first observed training point is retained. Subsequent curves use contiguous
100-record means; the recent summary uses the last 100 records. Diagnostics
retain their actual observations, including step zero, without averaging.
Very long runs therefore accumulate chart/diagnostic points in server memory.

The API exposes the exact recent update window and valid sample count for each
mean in `telemetry_window`. It also reports how many records have a known
profiling flag and how many were profiled. `freshness` identifies whether age
comes from a source timestamp or file modification time. A copied file's mtime
is file activity, not proof of recent training. Negative durations and non-finite
values are excluded from numeric means and remain missing.

State contract: `status` (`RUNNING`, `COMPLETE`, `PAUSED`, `FAILED`),
`target_updates`, `durable_checkpoint_update`. Completion is a producer
assertion, not inferred from a budget. ETA uses observed progress over at least
a minute, not log-reading speed. Write state atomically if adapting a producer.

## Diagnostic panels and model details

The monitor does not prescribe a particular research diagnostic or threshold.
For example, configure validation accuracy or representation rank:

```json
"diagnostics": [
  {"key": "validation_accuracy", "label": "Validation accuracy", "unit": "fraction"},
  {"key": "effective_rank", "label": "Effective rank", "help": "Your stated diagnostic definition"}
]
```

Log them with `monitor.diagnostic(update, metrics, healthy=None)`, or map an
existing diagnostic log with `diagnostic_fields`. Defaults are `step: update`,
`healthy: healthy`, and each metric at `summary.<key>`. `healthy` is an optional
producer assessment and does not control training. A numeric `reference` draws
a guide where it falls within the observed chart range. No universal threshold
or stop policy is imposed.

`details` supplies the architecture/training table: model, optimizer, precision,
batch/accumulation, data geometry, and whatever facts are useful for this run.
Only explicitly supplied scalars are shown. They are labelled configured
metadata, not a model architecture inferred by the dashboard.

## Light hooks

`Monitor` takes a JSON-serializable training config and stores its SHA-256
identity, not its contents. The output directory is dedicated monitoring
storage; a conflicting run/config is rejected. It never writes a training
checkpoint or scientific log owned by another logger.

`start_step` must reflect your durable resumed update. Use a new monitor output
directory if you rewind progress so the old and new trajectories are not mixed.
Place `Monitor.step(update)` around one whole optimizer update, including all
microbatches if using accumulation. `observation.record` accepts existing Python
scalars. A tensor is treated as unavailable; the hook never calls `.item()`.

`phase()` times explicit host boundaries on the main training thread. Nested
timers and asynchronous work can overlap, so the ordinary phase table presents
durations/call counts rather than claiming an additive critical-path partition.
Annotate worker-thread work in a separate producer if needed; these hooks do not
attribute it automatically.

`operation()` measures evaluation, checkpoint and journal work outside steps.
These cumulative costs remain visible without profiling. Include their
duration in your end-to-end throughput accounting rather than confusing the
update timer with complete training wall time.

Summary writes are rate-limited (`publish_seconds=5`); telemetry appends occur
each monitored update. `flush()` and `close()` publish final summaries.
Observation I/O/profiler failures are recorded without replacing training-loop
exceptions. Invalid hook usage is a programming error and raises immediately.
Profile failure disables further framework captures for that Monitor instance;
light monitoring continues.

## Connection to existing infrastructure

The server listens on loopback by default and needs no job-control privileges.
Use your SSH tunnel or authenticated network access; this server supplies no
application authentication/TLS. Monitoring routes are read-only: `/`,
`/api/metrics`, and `/api/dmon/raw`. The last route contains hardware samples,
not training log records.

Optional decision support adds `GET /api/decisions` and a guarded
`POST /api/proposals/review` for user decisions. This records consent only;
there is no training-command endpoint. See `decision-support.md` for its
single-user access assumptions and agent responsibilities.

GPU activity remains board-level activity; it is not automatically assigned to
one training job. Multiple jobs, missing hardware and unsupported counters
must remain visible limitations. The live GPU view is optional; CPU-side
monitoring and training panels continue when NVIDIA tools are absent.
