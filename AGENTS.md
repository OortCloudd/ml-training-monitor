# Connect the monitor to an ML training project

[SKILL.md](SKILL.md) is the unified entry. Performance investigation, runtime
inspection and MFU references are included in this repository; no separate
performance skill is required. Preserve every existing monitoring, profiling
and optional proposal-review capability when extending the integration.

You are the integration and optimization agent for the user's training environment.
Read [environment adaptation](docs/environment-adaptation.md) when installing or
changing that integration. Discover the actual infrastructure, implement the
necessary connections, and verify the measurements against their sources.
Preserve the dashboard and its collection capabilities. The same package serves
each installation; project paths, launchers and scheduler adapters belong locally.
An example workload is optional; if used, identify synthetic data explicitly.

For monitoring and optimization, read `docs/agent-access.md`. `/api/metrics`
provides the dashboard's measurements, timestamps, units, scopes and window
quality. Use your judgment to inspect bottlenecks and optimize the actual
training code. Ordinary monitoring imposes no optimization procedure.

Optional decision support requires explicit user opt-in. Read
`docs/decision-support.md` before using it. Submit concrete proposals through
`mlmonitor.proposals`; the user reviews them in the dashboard. Never approve
your own proposal or call the human review endpoint. Immediately before
execution, check the current approval, content digest, configuration context
and actual target revision. Stay within that scope. Enabling the mode is not
blanket permission to modify training.

## Where each project-specific decision belongs

| Project-specific item | Integration point |
| --- | --- |
| Runs, file locations, visible filesystems, physical GPU selection | Ignored `monitor.local.json` |
| Existing scheduler run/PID/GPU identity | Optional `runs[].runtime` JSON mapping, verified against the GPU's sole compute PID |
| Existing JSONL field names | `runs[].fields` and `runs[].files` |
| Completed update counter and resume position | `Monitor.step(update)` and constructor `start_step` |
| Input, forward, backward, optimizer and checking boundaries | `Monitor.phase(name)` in the existing loop |
| Existing scalar loss/gradient/timing measurements | `observation.record(...)` |
| Useful-model FLOP counts and the precision-matched hardware ceiling | `runs[].mfu`, mapped `fields.model_flops`, or optional `observation.record(model_flops=...)`; see `docs/mfu.md` |
| Checkpoint, evaluation and logging costs outside updates | `Monitor.operation(name)` |
| Durable checkpoint path, including terminal checkpoint | `Monitor.checkpoint(update, path, terminal=...)` AFTER the save |
| Model/optimizer facts intended for display | `details`; the raw training config is never exported |
| Diagnostic scalars and optional reference lines | `Monitor.diagnostic(...)` plus `runs[].diagnostics` |
| Optional framework profiling cadence | `profile_every` / `profile_steps`, disabled by default |
| Optional heavy profiling requests | `heavy_every` / `heavy_steps`, disabled by default |
| Scheduler, pause/resume, GPU ownership, replay command | User-owned adapter implementing `ready()` and `capture_context()` |

## Integration guidance

1. Read `docs/environment-adaptation.md` and `docs/integration.md`. Inspect the user's current logging
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
Keep a deployed checkout at its validated revision while preparing changes in a
separate development checkout/worktree; see `docs/unification.md`. An installed
skill and a running dashboard should not follow untested development edits.
Read `local/DEPLOYMENT.md` when present before changing an existing installation.

## Code map

- `mlmonitor/recorder.py`: light hooks and optional framework capture.
- `mlmonitor/reader.py`: existing-log adapters and histories.
- `mlmonitor/mfu.py`: shared MFU contract, aligned window calculation and provenance.
- `mlmonitor/config.py`: explicit deployment settings.
- `mlmonitor/step_attribution.py`: exclusive interval accounting.
- `mlmonitor/worker.py`: optional heavy capture worker and `CapturePlan`.
- `mlmonitor/import_ncu.py`, `gpu_efficiency.py`: raw metric import/validation.
- `mlmonitor/dmon_monitor.py`, `hardware.py`: live hardware collectors.
- `mlmonitor/server.py`, `index.html`: the existing dashboard panels.
- `mlmonitor/advice.py`, `proposals.py`: optional measured leads, proposal
  records and the agent interface; human review remains separate.
- `docs/performance-engineering.md`, `measurement.md`, `gpu-efficiency.md`:
  integrated investigation method; `scripts/summarize_trace.py` reads existing traces.

Run `python3 -m unittest discover -s tests -v`. Keep `local/`, runtime files,
traces, credentials, and private deployment settings out of commits. Publishing
requires the user's explicit request; installation alone does not authorize it.
