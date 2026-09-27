---
name: ml-training-monitor
description: Install and adapt training monitoring to the user's machines, GPUs and cluster workflows; track progress, calculate MFU, profile bottlenecks and validate optimizations. Use for agent-led integration and performance work while preserving the training recipe and existing job control.
---

# ML Training Monitor

One entry for the dashboard, training telemetry, runtime inspection, MFU,
profiling and evidence-based optimization. This skill equips a coding agent to
install and adapt the supplied tools to its user's actual training environment,
then investigate and improve training within the requested scope. The
implementation is in `mlmonitor/`; preserve its panels and collection capabilities
as you connect them. Read [AGENTS.md](AGENTS.md) for integration points.

For a new installation or infrastructure change, start with
[environment adaptation](docs/environment-adaptation.md). Discover the available
machines, allocated GPUs, launchers and existing logs; implement the necessary
local mappings or adapters and verify their real output. Multi-host or cluster
integration is work for the agent to perform and qualify: the supplied collectors
start on one host. A demonstration run is optional, not an installation prerequisite.

## Choose the work the user asked for

| Request | Start here | Result |
| --- | --- | --- |
| Current progress, ETA, hardware or training status | [Agent access](docs/agent-access.md), [runtime inspection](docs/runtime-inspection.md) | Timestamped answer from the actual run and measurement window |
| Install, adapt to GPUs/clusters, or migrate a dashboard | [Environment adaptation](docs/environment-adaptation.md), [integration](docs/integration.md), [unification](docs/unification.md) | Working connection to the user's environment, with validated scope and preserved panels |
| Useful GPU compute throughput / MFU | [MFU](docs/mfu.md) | Shared `runs[].mfu` result for the dashboard and agents, with its evidence and limits |
| Slow training or an optimization request | [Performance engineering](docs/performance-engineering.md) | Bottleneck evidence, bounded intervention and appropriate equivalence checks |
| Timing or attribution ambiguity | [Measurement](docs/measurement.md) | Correct boundaries, overlap accounting and launch parity |
| GPU efficiency, Nsight, roofline or periodic captures | [GPU efficiency](docs/gpu-efficiency.md), [profiling](docs/profiling.md) | Supported counters and dated, scoped captures |
| User-enabled proposal review | [Decision support](docs/decision-support.md) | Concrete proposal and user review; preserve the current review mechanism |

Load only the references needed for the request. Ordinary monitoring does not
require running an optimization checklist, starting a profiler, or enabling
proposal review. In an authorized optimization task, follow the performance
routine through validation and the requested stopping point.

## Shared rules

Identify the actual run, configuration, source of each measurement and timing
boundary. The dashboard and agent consume the same `/api/metrics` contract.
Keep measured data, analytical estimates, bounds and interpretation distinct.
Missing values stay missing; stale profiles retain their dates. High GPU activity
does not prove high MFU, bandwidth saturation, useful throughput or model quality.

Connect existing logs before adding hooks. Preserve model/data/optimizer/RNG,
accumulation, checkpoint and resume semantics. Do not change the scientific
recipe to improve an efficiency percentage. Explicit hooks belong in the
existing loop; project launchers, GPU leases and recovery stay in project
adapters. Heavy profiling and training mutation remain within the user's scope.

For MFU, use matched useful FLOPs and elapsed time over the same updates, plus
the assigned GPU count and precision/sparsity-matched peak. State approximations
and excluded operations. Prefer per-update counts for variable workloads;
configured estimates or bounds remain visibly labelled. When periodic MFU is requested,
attach it to the existing profiling cadence and display a dated capture; do not
extrapolate it into a live percentage or launch a separate profiling scheduler.
Profile overhead must stay out of the timing reference. Recompute the ratio
from totals; never substitute utilization/SOL or invent missing token counts.

Keep private deployment configuration, local paths, logs, patient data and
credentials outside tracked examples. Before material writes inspect space on
the actual destination filesystem. Preserve the repository's existing optional
decision support and all dashboard/collector capabilities when extending it.

## Implementation and validation

Run the existing dashboard with `python3 -m mlmonitor --config monitor.local.json`;
use `--check` first. [README.md](README.md) gives the complete starting commands.
`Monitor.step/phase/operation/checkpoint/diagnostic` provide the existing hooks.
MFU integrates through run configuration, mapped logs or
`observation.record(model_flops=...)`; [docs/mfu.md](docs/mfu.md) defines the contract.

For an existing PyTorch Chrome trace with `perf::step` ranges, use
`python3 scripts/summarize_trace.py TRACE.json --output summary.json`. It computes
per-device interval unions, not FLOP efficiency or occupancy. The scheduled
recorder and Nsight worker remain the production profiling implementation;
this helper is an offline importer, not a second scheduler.

Run `python3 -m unittest discover -s tests -v` and verify the affected actual
producer-to-API-to-dashboard path. Report what changed, evidence, limitations
and any remaining migration step. Keep public publication separate from local
preparation unless the user requested publishing.
