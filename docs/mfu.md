# Model FLOPs Utilization

MFU is useful model forward/backward throughput relative to the theoretical
compute capacity assigned to a run. It is distinct from GPU activity, SOL,
occupancy, and model quality. The dashboard and agents share `runs[].mfu` in
`/api/metrics`; there is no separate agent-only calculation.

`MFU (%) = 100 × sum(useful model FLOPs) / sum(step seconds) / total peak FLOP/s`.

Count encoder, decoder/projector, attention, all views and all accumulated
microbatches of an optimizer update. For variable shapes, account for actual
visible/selected/target tokens. A language model's `6 × parameters × tokens`
shortcut is not a universal vision-model formula. Use a declared multiply-add
convention (usually two FLOPs) and a documented backward approximation.
Exclude activation/gradient-cache recomputation from useful FLOPs, but retain
its time in the denominator. State omitted norms, activations, optimizer and
other operations. No model-specific FLOP estimator is silently assumed.

Use the actual GPU model/count, precision and sparsity. Dense workloads use
dense peaks, even if a larger sparse number appears in marketing. The current
contract covers a homogeneous GPU allocation; heterogeneous per-device peaks
and distributed per-rank aggregation need an explicit project adapter.

Definition: [PaLM, section 4 and Appendix B](https://arxiv.org/html/2204.02311v5).
Manufacturers define hardware ceilings; their units and operation convention
must match the chosen useful-operation count. Record the source and date.

## Configure a run

Optional `runs[].mfu` requires these fields:

| Field | Meaning |
| --- | --- |
| `config_sha256` | Exact configuration fingerprint in the run binding |
| `gpu_model`, `gpu_count` | Explicit homogeneous allocation for this run |
| `precision`, `sparsity` | Arithmetic precision; `dense` or `structured_sparse` |
| `peak_tflops_per_gpu` | Decimal theoretical TFLOP/s for that arithmetic |
| `peak_source` | Manufacturer reference, verification date and peak convention |
| `flop_source` | Formula/profile provenance and coverage |
| `timing_scope` | Exact logged elapsed-time boundary |
| `mode` | `per_step`, `constant`, or `bounds` |

Additional fields: `dataset_manifest_sha256` (when FLOPs depend on that dataset),
`assumptions` (list of strings), `flops_per_update` for constant mode and
`flops_per_update_bounds: [lower, upper]` for bounds mode. Constant mode may
also supply bounds containing its point estimate. A configuration mismatch,
known environment mismatch or missing timing produces an unavailable value.
Declared allocation/ceilings are configured metadata, not a hardware measurement.

Choose a mode based on the evidence:

- **Per step:** map `fields.model_flops` to existing useful-operation counts.
  The default field is `model_flops`. The monitor sums counts and times from
  the same records; any missing/nonpositive value makes that window unavailable.
- **Constant:** supply a justified fixed or representative per-update budget.
  A corpus-weighted shape estimate must say that actual batches may differ.
  This remains an analytical estimate, not a hardware FLOP counter.
- **Bounds:** provide justified minimum/maximum work when variable token counts
  are unavailable. No central value is generated. These are architectural
  bounds under the stated approximation, not confidence intervals.

If existing loop hooks are already integrated, an existing Python scalar can
be passed as `observation.record(model_flops=useful_flops_this_update)` together
with other metrics. Tensors are never converted with `.item()`. This field is
optional and does not change existing logger output when omitted. Register
the matching MFU contract in the dashboard configuration separately.

## Window and interpretation

The API reports the last 100 records (or fewer initially), first/last update,
valid counts, summed seconds, observed profiling flags, stale/completed state,
useful TFLOP/s and configured peak provenance. Backlog reading is explicitly
unavailable until the latest records are reached. Missing is never zero.

The calculation is a ratio of totals, not a mean of step percentages. The
window follows the logger's timing boundary: an update timer excluding external
checkpoint/evaluation time is not end-to-end wall-clock MFU. Profiled updates
remain included and counted, matching the dashboard's existing mean window;
unknown flags are not relabelled as unprofiled.

Never clamp inconsistent point estimates above 100%. Check count conventions,
batch/accumulation, duplicated forward or recomputation, timing synchronization,
GPU count, precision and sparsity. An upper work bound may exceed the hardware
ceiling without proving that the actual workload did; a lower bound above the
ceiling is inconsistent. Low MFU alone does not diagnose its cause or authorize
changing the scientific recipe. Use the timeline and controlled comparisons
in [performance engineering](performance-engineering.md).

## Periodic capture mode

Use `mode: "capture"` and `files.mfu_capture` for a dated measurement produced
by the existing scheduled profiler. The contract also requires the unique
`gpu_uuids` and an `estimator_sha256`. The API validates the recorded allocation,
configuration, precision, counted work and unprofiled timing window before
publishing a percentage. It does not extrapolate a last capture into live MFU.
The capture must also match the native `run_id` in `run_bindings.json`, or the
registered run ID when that binding is absent. A dashboard display alias can
therefore retain the original producer identity. Matching configuration and
GPU alone does not make a capture from a different run applicable.

The project adapter counts CUDA matrix/convolution/attention operations during
the profiling update, executing each operator unchanged. A lightweight estimator
uses the actual input/output tensor shapes on the preceding unprofiled updates.
Its count must match the dispatched operator count. Numerator and denominator
come from the same reference updates. Profiling time is never substituted for
ordinary step time. Recomputation is separate from useful model work; omitted
operation families and the counting convention stay explicit.

The integration must reuse the existing profiler cadence and worker, not create
a polling-triggered profiler. A local deployment may select step 50 then each
30,000 steps. A one-off installation check is explicitly tagged and does not
change the recurring cadence or claim to reconstruct an earlier step.

`confidence_interval` concerns local work/time variability, conditional on
local stationarity. The current moving-block bootstrap uses 2,000 replicates,
blocks of 5–10 updates and at least 50 reference updates. A greater-than-10%
rate change between half-windows withholds the interval. These are explicit
engineering defaults, not a universal guarantee of statistical coverage.
Intervals exclude systematic counting/coverage and hardware-ceiling uncertainty.
The minimum/maximum observed step rates are descriptive ranges, never confidence
intervals. A shorter early window still has a dated observed ratio, with no
invented interval.

For distributed data parallelism, the adapter must provide unique rank-local
work from every allocated rank for the same run/config/update, and one common
coordinator update duration including collective waiting. Sum work once and
use that one duration; do not sum rank durations, average rank percentages,
or combine independent training runs. `reduce_data_parallel_step` validates
this contract. Tensor/pipeline-parallel work ownership needs a specific adapter;
the utility does not infer it. Local single-GPU validation is not a claim of
validation on a large distributed cluster.
