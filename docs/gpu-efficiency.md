# GPU activity, efficiency and temporal evidence

Use this reference when utilization is high but throughput is disappointing, when comparing accelerators, or when exposing performance metrics in a dashboard. GPU busy percentage is not FLOP efficiency and is not a bottleneck diagnosis.

## Start with a time window

For NVIDIA hardware, a bounded read-only observation can use:

```bash
nvidia-smi dmon -i 0,1 -s pucm -d 1 -c 120 -o DT
```

Adjust device selection to the actual run. Record the start/end time, sample interval, count, missing samples, active job and its progress over the same interval. Report mean, median, p95 and range; correlate dips with input waits, synchronization, checkpointing and evaluation. A single screenshot is not a representative observation. For continuous dashboards, retain timestamped samples and label the window and its completeness. A sampling mean is not a continuous integral when intervals are irregular; expose gaps.

`p` = power/temperatures, `u` = activity, `c` = clocks, `m` = framebuffer/BAR1 capacity used. In ordinary dmon, `sm` is GPU utilization/activity, **not achieved SM occupancy**. Its `mem` is the fraction of the sampling interval in which device memory was read/written, **not bytes/s or percentage of peak memory bandwidth**. VRAM capacity used is a third distinct quantity. Do not derive DRAM throughput by multiplying `mem` by the GPU's peak bandwidth. Power and clocks are supporting evidence; high power does not prove useful work and a low clock alone does not prove thermal throttling.

If available, GPM/DCGM can provide additional sampled metrics. Discover support on the actual GPU/driver; a listed CLI metric is not proof of support. `-`, unsupported and denied counters must remain missing, never zero. Profiling/GPM percentages have their own definitions and normalization: they are not interchangeable with dmon's ordinary activity columns.

## Four possible high-activity regimes

These are hypotheses that may coexist across kernels or phases, not mutually exclusive whole-run labels.

| Hypothesis | Evidence required | Common wrong inference |
|---|---|---|
| Compute throughput limit | Relevant arithmetic/Tensor pipeline throughput near its sustained ceiling, appropriate-precision roofline, sufficient work; substantial critical-path time in those kernels | High SM activity or high overall SOL compute means Tensor Cores are saturated |
| Memory throughput limit | Inspect DRAM, L2 and L1/TEX separately; bytes/s and arithmetic intensity near the matching bandwidth roof, access efficiency | High memory activity or allocated VRAM means bandwidth saturation |
| Many small inefficient kernels | Systems/PyTorch timeline: durations, launch rate, gaps, sizes and aggregate share; NCU for selected representative kernels | GPU continuously busy means each kernel fills the device |
| Latency/occupancy/stall limit | Achieved versus theoretical occupancy, eligible warps, issue activity and limiting resources/stalls; launch geometry, registers/shared memory | Low occupancy is necessarily bad, or high occupancy guarantees performance |

Speed of Light (SOL) gives achieved compute/memory throughput relative to supported theoretical/sustained unit ceilings. Its aggregates can be determined by the busiest constituent subunit; inspect the breakdown rather than translating a percentage into model FLOP utilization. Tensor pipeline activity is not automatically the fraction of useful BF16 FLOPs delivered. Occupancy describes resident warps, not whether they issue useful instructions. Interpret stall reasons primarily where the schedulers fail to issue; a large stall percentage without that context is not actionable.

## Nsight workflow

1. Identify tools, GPU architecture/driver and access to performance counters. Check installed `ncu --version`, `ncu --list-sections`, `ncu --query-metrics` as applicable. Section/metric names vary by release and GPU.
2. Inspect `/proc/driver/nvidia/params` for `RmProfilingAdminOnly`. A value of 1 normally requires an administrator/capability for performance counters. Do not alter/reload the driver or stop unrelated jobs to acquire access. Prepare the capture and use an authorized administrator-assisted replay if available; otherwise report the concrete restriction.
3. Use a Systems/PyTorch trace to locate the important phases first. Profile a bounded warmed isolated replay and selected representative expensive kernel families, including fallbacks. A GEMM alone cannot characterize an entire mixed workload.
4. Start with relevant sections discovered in step 1: `SpeedOfLight`, `ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis`, `Occupancy`, `SchedulerStats`, `WarpStateStats`, and the appropriate roofline section. Recent versions include `SpeedOfLight_HierarchicalTensorRooflineChart`; support must be verified. Do not blindly collect `--set full` on a complete training run.
5. Record replay mode, passes, cache/clock control, warmup, shapes, precision, kernel/step coverage, run/config identity and capture date. Nsight may serialize execution, replay kernels and change cache state; its captured timings are not production throughput. Do not stack two CUPTI profilers and assume measurements remain valid.
6. Attach measured counters to the selected kernel/range. Diagnose its bottleneck and establish its share of representative unprofiled step time before extrapolating to the run. A duration-weighted average of selected-kernel SOL percentages is not a hardware-counter measurement of whole-run utilization.

Useful metric families (examples, query on the installed device):
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `lts__throughput.avg.pct_of_peak_sustained_elapsed`
- `l1tex__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `smsp__warps_eligible.avg.per_cycle_active`
- `smsp__issue_active.avg.pct_of_peak_sustained_active`

Preserve each raw metric name, unit and active/elapsed denominator in an export. Never silently substitute one metric for another when a name is unavailable.

## Periodic profiling during long runs

For an authorized recurring performance review, choose a capture cadence in
completed optimizer updates or elapsed time. For example, every 30,000 optimizer
updates may fit a long run with checkpoints at those milestones; it is an
example, not a default requirement. Define the unit explicitly: accumulated
microbatches are not optimizer updates. This skill supplies the workflow, not an
installed scheduler.

Keep lightweight telemetry between captures. At a due milestone, use a short
warmed Systems/framework timeline, then selected Nsight Compute counters only
where they answer an unresolved question. Follow the bounded-replay and recovery
procedure in `SKILL.md` if capture requires interrupting training. Record the
scheduled milestone and actual captured checkpoint/update; a late capture must
not be labelled as an earlier measurement.

For each capture, retain run/revision/config identity, checkpoint/update, tool
versions, workload shapes and precision, coverage, unprofiled throughput,
profiling overhead, and the largest remaining critical-path costs. Compare with
the previous compatible capture; flag configuration changes that invalidate a
direct comparison. Reprofile after a material runtime change or unexplained
throughput regression when within scope, rather than waiting for the next
milestone.

If automating this workflow, allow only one capture per run/milestone, prevent
overlapping profilers, check space and set retention and time limits. Record
failed or skipped captures and use a bounded retry policy; unavailable counters
must not trigger an endless pause/retry cycle. Verify that production resumes
and progresses after any authorized interruption, including a failed capture.

## Roofline and dashboards

Roofline compares measured throughput with arithmetic intensity (operations/byte) and compute/bandwidth ceilings. Use the same precision and operation-count convention for the point and ceiling, and the same memory level for byte traffic and its roof. A scalar FP32 roofline does not characterize a BF16 Tensor Core workload. State FMA/MMA counting, sparsity assumptions and decimal/binary units. A point far below both roofs indicates headroom, not uniquely its cause.

Separate:
- **Live telemetry:** timestamped activity, power, clocks, capacity and supported passive counters.
- **Profile captures:** dated SOL compute/memory/Tensor/occupancy metrics and roofline for identified kernels/ranges.
- **Inference:** the interpretation and confidence, including unresolved bottlenecks.

Show unavailable counters explicitly. Never draw a measured roofline point from activity %, theoretical FLOPs or guessed byte traffic. Show the tool/version, hardware, precision, scope and configuration mismatch for historical captures. Do not launch profilers from periodic page refreshes. No synthetic fixture measurements should be published as real GPU data.

## Sources

- NVIDIA [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): SOL sections, throughput metrics, roofline, occupancy, scheduler stalls and replay effects.
- NVIDIA [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html): tool/section/metric discovery and bounded captures.
- NVIDIA [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html): activity definitions, dmon sampling and unsupported metrics.

Guide and nvidia-smi documentation checked directly on 2026-09-24. These sources explain metrics; they do not establish the bottleneck of our runs.
