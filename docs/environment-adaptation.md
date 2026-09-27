# Adapt the monitor to the user's environment

This skill is for coding agents. Use the user's actual training and available
infrastructure to connect the supplied dashboard, collectors and profiling
workflow. A demonstration run is optional; synthetic data only checks the
integration and must be labelled synthetic. It proves no training speedup.

## Discover enough to choose the integration

Read the project's launch configuration and current logs, then inspect the
relevant running processes and scheduler allocations through the user's existing
access. Establish the hosts/clusters involved, framework environment, GPU types
and assignments, container/device visibility, shared storage, and log writers.
Available hardware means hardware the project can allocate, not every GPU an
agent can discover. Record unknown or unreachable parts without inventing data.

Determine one optimizer update, accumulation, local versus global batch, and
which rank (distributed training process) reports completed progress. Identify
checkpoint/resume boundaries before attaching profiling. Retain these facts in
the project's local integration notes so another agent needs no chat history.

## Choose the smallest working connection

| Environment | Shipped behavior | Agent adaptation when needed |
| --- | --- | --- |
| One host, one or several GPUs | Linux host metrics, local NVIDIA collectors, explicitly registered run logs | Map paths, scalar fields and optional scheduler identity in `monitor.local.json` |
| Several hosts with readable run logs | One dashboard can read those logs; its hardware panels still measure the dashboard host | Identify each log's origin; use a monitor on each relevant host or implement a project-specific remote collection bridge |
| Distributed run across GPUs/hosts | Progress from one canonical writer can be displayed | Establish global measurement units, rank/device identity and any additional attribution or aggregation |
| Several clusters or mixed GPU types | No built-in cluster discovery, remote transport or federation | Connect the existing site scheduler/access and extend collection/presentation only as the requested view requires |

The agent is expected to implement needed adaptations; an unsupported layout
is an engineering task, not a reason to substitute a reduced dashboard. Reuse
the existing panels and contracts. Keep site paths, transport code and scheduler
adapters under ignored `local/`, with deployment settings in the ignored
`monitor.local.json`. A reusable extension can live in the package with tests;
site credentials and private records cannot become public defaults.

For remote work, first choose whether separate host dashboards satisfy the
request. A consolidated view requires an explicit bridge and corresponding
schema/UI changes: there is no `remote_hosts` configuration or automatic merge.
Use the site's existing access and job lifecycle; do not replace its scheduler.

## Preserve identity and measurement meaning

Use stable cluster, host, run/attempt and physical device UUID identities in a
bridge; a PID, `GPU 0`, or scheduler job number alone is not globally unique.
Run IDs accepted by this package contain 1–64 letters, digits, underscores or
hyphens. A local identity map can translate a richer site identity into those
IDs. Keep an allocation/restart history when a distributed job changes ranks or
devices; never combine a rewound attempt with its predecessor's trajectory.

The existing `runs[].runtime` mapping reads one JSON record with run ID, PID,
GPU UUID and status. `runtime.assign_runs()` verifies that this PID is the GPU's
sole compute process. It represents one device per run mapping, not a distributed
allocation. For DDP or other multi-process layouts, preserve the host hardware
view and unidentified associations until an extension verifies the complete
allocation. Copying a scheduler job's name onto every busy GPU is not verification.

Keep one canonical progress stream per logical run. Rank logs must not all append
the same updates into that stream. Define whether loss, samples/tokens and FLOPs
are rank-local or global; reduce them according to their actual semantics. Count
each contribution once, including model-parallel shards and replicated metrics.
Use elapsed wall time for the same completed global work, not a sum of concurrent
rank times. Record host clock differences and transport delay; retain producer
timestamps, collection timestamps and stale/missing state when bridging data.

MFU (model FLOPs utilization) compares useful computation with the assigned
hardware's theoretical arithmetic capacity over that same elapsed window.
The current contract uses `gpu_count * peak_tflops_per_gpu`: it assumes a common
precision/sparsity-matched peak per GPU. Heterogeneous allocations need an
extension with explicit per-device ceilings and their sum, or separately scoped
homogeneous groups with known work counts. Do not insert one GPU model's peak
for a mixed allocation or average device percentages. Leave MFU unavailable
while the numerator, timing or allocation is unknown; all other panels remain.

## Extension points

| Code / contract | What to connect or extend |
| --- | --- |
| `mlmonitor/config.py`, `reader.py` | `runs[].files`, `fields`, diagnostic mappings, canonical scalar JSONL and atomic state; see [integration](integration.md) |
| `mlmonitor/hardware.py`, `dmon_monitor.py` | Host-local collection; a remote adapter must preserve source host/device, units and timestamps |
| `mlmonitor/server.py:Sampler`, `index.html` | Snapshot assembly, history and display; GPU windows currently key by local index, so namespace these together for federation |
| `mlmonitor/runtime.py:assign_runs` | Verified allocation/compute-process association; extend the one-record mapping for distributed ownership |
| `mlmonitor/mfu.py`, `mfu_capture.py` | Useful-operation accounting and matching hardware/timing provenance; extend validation and display with a new denominator |
| `mlmonitor/recorder.py:Monitor`, `TorchCapture` | Existing-loop hooks and selected-device framework capture; choose writers/capture ranks explicitly |
| `mlmonitor/worker.py:CapturePlan` | Local checkpoint replay using `ready`, `capture_context`, and recovery; see [profiling](profiling.md) |
| `/api/metrics` (`ml-monitor-snapshot-v1`) | Shared agent/dashboard output; see [agent access](agent-access.md) for semantics |

There is no dynamic collector-plugin loader. Adding remote hardware requires
wiring the bridge into collection and display, not merely writing an adapter
file. Preserve the existing local path and make any changed API contract
explicit. Profiling orchestration belongs to the project's existing launcher;
the shipped Nsight worker is a single-GPU capture, not a distributed launcher.

## Show that the adaptation works

Validate the changed path against its actual producer: source update and unit →
API value/window → dashboard panel. For distributed work, show which canonical
writer and allocation supplied that value. If implementing federation, exercise
duplicate local GPU indices, a missing host, stale samples and an allocation
change so no hardware or progress silently changes owner. Add focused tests for
new aggregation/identity logic and run the existing suite after package edits.

An installation can be validated from existing logs without starting a training
job. Report exactly which paths were exercised, which require the user's runtime,
and any synthetic fixtures used. For an authorized optimization task, continue
with [performance engineering](performance-engineering.md): profile, test a
bounded change, verify equivalence, measure and reprofile the integrated path.
The final handoff contains the local config/adapter locations, start/read
commands, verified data sources and remaining gaps, without relying on this
skill's original deployment or conversation history.
