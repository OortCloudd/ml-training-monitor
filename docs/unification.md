# One monitor, one skill entry

ML Training Monitor is the maintained entry for the dashboard, runtime
inspection, MFU, profiling and performance engineering. Its existing Python
package remains the implementation. Guidance extends it rather than wrapping
it in another dashboard or replacing its recorder/worker/review workflow.

## Preservation and integration map

| Existing capability | Unified home | Preservation contract |
| --- | --- | --- |
| Run curves, ETA, gradients, checkpoint state | `reader.py`, `index.html`, local run configuration | Keep history and missing-data semantics |
| CPU/RAM/filesystems and GPU activity/power/clocks | `hardware.py`, `dmon_monitor.py`, `gpu_efficiency.py` | Keep provenance, windows, gaps and unsupported counters |
| Model details and configurable diagnostics | Existing run `details`, `diagnostics`, `diagnostic_fields` | Preserve project definitions; no universal clinical thresholds |
| Light phases and outside-update operations | Existing `Monitor.phase/operation` and file adapters | No extra synchronization or changed training loop semantics |
| Scheduled framework captures and their history | Existing recorder and capture loader | Preserve optional opt-in, cadence and dated identity |
| Nsight import/replay/recovery | Existing worker, importer and project adapter | No additional worker or implicit capture on refresh |
| Measured leads and human proposal review | Existing advice/proposals and decision-support panel | Preserve disabled-by-default mode and every review/outcome state |
| Performance diagnosis and equivalence testing | `docs/performance-engineering.md`, `measurement.md`, `gpu-efficiency.md` | Bring the full existing method; apply it to optimization requests |
| Offline trace analysis | `scripts/summarize_trace.py` and its original behavioral tests | Keep interval unions and per-device attribution |
| Runtime status inspection | `docs/runtime-inspection.md` | Read-only inspection unless repair is authorized |
| MFU | `mlmonitor/mfu.py`, shared API/UI, `docs/mfu.md` | Same calculation for agents and dashboard; estimates/bounds stay explicit |
| Discovery and task routing | Root `SKILL.md`, `agents/openai.yaml` | One canonical skill; older names may redirect |

The integration baseline is `ml-training-monitor` commit
`981b1f1f47ea8dd7c34e63681b7542b0d1e834a1`. No baseline source file or capability
is removed. The performance method, references and offline helper originate
from `ml-performance-engineering` commit
`daf55cf` (Apache-2.0); only relative links and the unified routing are adapted.
The original trace tests remain in the combined test suite. Project-specific
MFU formulas, manifests and private paths remain in local adapters/configuration,
not generic defaults.

## Migrating an existing local dashboard

1. Preserve the current service definition and source revision. Inventory its
   run IDs, file mappings, diagnostics, architecture provenance, profiles,
   historical captures and any caller of its API.
2. Register those sources through an ignored local configuration. Reference
   existing files directly; do not copy datasets or rewrite training journals.
   Preserve dataset/config bindings for analytical MFU profiles and identify
   actual timing boundaries. Keep optional decision support in its current mode.
3. Start the candidate on a separate loopback port. It remains an ordinary
   read-only consumer. Compare common update windows with the existing reader:
   steps, loss, timings, gradients, checkpoint state, diagnostics, profiles and
   MFU. Check unavailable/mismatch states and all UI panels on desktop/mobile.
4. Prepare the service's new working directory and command plus an exact
   rollback. Change only the dashboard service when the requested work includes
   activation. Training processes, schedulers, checkpoint paths and scientific
   configs stay outside a dashboard migration.
5. Install the root skill from the validated deployed checkout. Turn an older skill
   name into a small compatibility redirect once its method and helpers are
   represented here. Preserve its history; do not continue editing two methods.
6. Verify the original dashboard address, API consumers and run progress after
   activation. Retire a legacy implementation only after verified parity and
   the user's requested scope allows retirement; preserving it for rollback is
   not a second maintained product.

Preparation, activation and public publication are distinct states. Record the
actual state in the local handoff. A prepared configuration or passing test is
not a claim that the live service has been migrated. Public examples contain
no site paths, patient data, training logs, private addresses or credentials.

## Validation

Run `python3 -m unittest discover -s tests -v`; use a training environment with
PyTorch already installed for the optional CPU model/optimizer/RNG parity test.
Validate the root skill frontmatter and verify local Markdown links. Compare
the baseline file inventory to ensure no existing file disappeared. Exercise
the real read-only producer-to-API-to-UI path separately from synthetic tests.

## Deployment and development

Keep the service and installed skill on an explicit validated revision. Use a
separate checkout or Git worktree to prepare public updates and project-specific
extensions. Both share the same package and upstream history; deployment does
not need its own implementation or a continuously diverging branch.

Keep private configuration and adapters in ignored local storage. A Git worktree
does not copy ignored files, so reference the existing configuration deliberately
when validating a candidate; do not assume its absence means there are no runs.
Record the deployed revision, service command, configuration location and prior
revision in ignored `local/DEPLOYMENT.md` so the next agent can update or roll
back the correct installation.

Before promotion, test the candidate and check compatibility with actual mapped
logs, captures and API consumers. An authorized update changes the deployed
revision deliberately, without following arbitrary development edits or pulls.
Runtime/configuration changes require the appropriate dashboard restart and live
verification; documentation-only changes with byte-identical runtime code do
not require restarting training or the dashboard. Preserve the existing access
address, private settings, history and project-owned profiling workers.
