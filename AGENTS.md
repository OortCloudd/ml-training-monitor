# Connect the monitor to an ML training project

The primary user is an ML practitioner who wants to understand their running
training. Preserve the dashboard and its collection capabilities. Your role is
to connect the user's workflow to the integration points below.

For monitoring and optimization, read `docs/agent-access.md`. `/api/metrics`
provides the dashboard's measurements, timestamps, units, scopes and window
quality. Use your judgment to inspect bottlenecks and optimize the actual
training code. No diagnosis classifier, experiment gate or fixed decision
procedure is imposed by this project.

If optional decision support is added, require explicit user opt-in and retain
the user's final decision on each concrete proposed intervention. Distinguish
proposal preparation from authorization to execute it. The current version
provides monitoring and data access; this optional layer is not implemented.

## Where each project-specific decision belongs

| Project-specific item | Integration point |
| --- | --- |
| Runs, file locations, visible filesystems, physical GPU selection | Ignored `monitor.local.json` |
| Existing JSONL field names | `runs[].fields` and `runs[].files` |
| Completed update counter and resume position | `Monitor.step(update)` and constructor `start_step` |
| Input, forward, backward, optimizer and checking boundaries | `Monitor.phase(name)` in the existing loop |
| Existing scalar loss/gradient/timing measurements | `observation.record(...)` |
| Checkpoint, evaluation and logging costs outside updates | `Monitor.operation(name)` |
| Durable checkpoint path, including terminal checkpoint | `Monitor.checkpoint(update, path, terminal=...)` AFTER the save |
| Model/optimizer facts intended for display | `details`; the raw training config is never exported |
| Diagnostic scalars and optional reference lines | `Monitor.diagnostic(...)` plus `runs[].diagnostics` |
| Optional framework profiling cadence | `profile_every` / `profile_steps`, disabled by default |
| Optional heavy profiling requests | `heavy_every` / `heavy_steps`, disabled by default |
| Scheduler, pause/resume, GPU ownership, replay command | User-owned adapter implementing `ready()` and `capture_context()` |

## Integration guidance

1. Read `README.md` and `docs/integration.md`. Inspect the user's current logging
   and training boundaries before editing. Establish what one update means,
   including gradient accumulation, and which rank writes progress.
2. Connect existing logs first. Configure scalar fields and missing values; do
   not copy private records or entire configs into documentation or commits.
3. If the user wants phase/outside-step timing, add the explicit hooks at the
   existing boundaries. Do not monkeypatch framework methods, replace the
   training loop, alter tensor order, or add synchronization in light mode.
4. Validate one actual producer-to-dashboard path. Match the displayed update
   and measurements to the source; check that missing profiles do not disable
   ordinary monitoring. Check model/optimizer/RNG parity when editing a loop.
5. Add profiling only within the requested scope. Read `docs/profiling.md`.
   The supplied worker already handles collection/import and deduplication;
   adapt only the user's readiness, checkpoint, device and recovery boundaries.

Config changes need a dashboard restart. Heavy profiling never follows from a
page refresh or passive-monitoring request. Run services, checkpoint formats,
source budgets, and terminal-step replay semantics belong to the user's project.
Do not bring in this monitor's original research deployment assumptions.

## Code map

- `mlmonitor/recorder.py`: light hooks and optional framework capture.
- `mlmonitor/reader.py`: existing-log adapters and histories.
- `mlmonitor/config.py`: explicit deployment settings.
- `mlmonitor/step_attribution.py`: exclusive interval accounting.
- `mlmonitor/worker.py`: optional heavy capture worker and `CapturePlan`.
- `mlmonitor/import_ncu.py`, `gpu_efficiency.py`: raw metric import/validation.
- `mlmonitor/dmon_monitor.py`, `hardware.py`: live hardware collectors.
- `mlmonitor/server.py`, `index.html`: the existing dashboard panels.

Run `python3 -m unittest discover -s tests -v`. Keep `local/`, runtime files,
traces, credentials, and private deployment settings out of commits. No
publication is authorized by these instructions.
