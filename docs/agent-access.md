# Access for an ML engineer or coding agent

The dashboard is a monitoring instrument. Use the measurements to understand
training, investigate code and choose optimizations. The agent owns the
reasoning, priorities and experiments. Ordinary monitoring does not impose an
optimization procedure.

[Optional decision support](decision-support.md) adds measured leads and
concrete proposals for user review. It requires explicit opt-in; the user
retains the final decision for each intervention. Agents submit proposals and
read approval records, but must not approve themselves. The dashboard never
executes training commands.

## Read the same data as the dashboard

```bash
curl --fail http://127.0.0.1:8790/api/metrics
curl --fail http://127.0.0.1:8790/api/dmon/raw
```

These read-only endpoints never start a profiler or change training. The first
returns `ml-monitor-snapshot-v1`; the second returns rolling hardware samples.

| Data | Where to read it |
| --- | --- |
| Current hardware and time series | `hardware`, `history` |
| Actual collection duration and requested cadence | `collection`, `interval` |
| GPU window boundaries, count, gaps and per-metric valid sample counts | `performance.windows`, `performance.dmon_live` |
| Live units, source fields and metric meanings | `performance.live_metric_definitions` |
| Run progress, loss, step time and gradients | `runs[]` |
| Exact recent update window and valid counts for displayed means | `runs[].telemetry_window` |
| Timestamp provenance, age and reported clock offset | `runs[].freshness` |
| Inclusive phase timers and checkpoint/evaluation costs | `runs[].light_profile`, `runs[].operations` |
| Dated framework profiles and their history | `runs[].bottleneck`, `runs[].profile_history` |
| Dated Nsight counters and their definitions | `performance.captures`, `performance.metric_definitions` |

Missing values are null, never substituted with zero. A partial window retains
its actual sample count and span. Statistics are sample-based, not a continuous
time integral. Hardware is board-level; training ownership is not inferred.
Source timestamps and file modification times are distinguished. Profiled
updates remain in live timing means, with their known count exposed; an absent
profiling flag is not interpreted as an unprofiled update.

Nsight and framework captures are dated observations, not real-time counters.
Their configuration identity and capture update remain attached. GPU activity,
memory activity, VRAM use, bandwidth and arithmetic throughput have different
meanings; the source definitions are included in the API.

## Brief optimization method

Read the current measurements, relate expensive regions to the actual training
code, and choose a useful experiment. Add a targeted capture when it helps.
Compare useful throughput on comparable work and check the behavior affected
by the change; remeasure after integration. The agent decides what to try,
what is worth keeping and when to stop, within the user's requested scope.

The [ML Performance Engineering skill](https://github.com/OortCloudd/ml-performance-engineering)
provides more methodological detail if wanted. It is not required to access the
dashboard or integrate the monitor.
