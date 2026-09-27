# Validation records

## Agent adaptation update — 27 September 2026

- **70 tests passed**, including the CPU PyTorch model/optimizer/RNG/loss parity
  test, using Python 3.12 and an existing PyTorch 2.6.0 environment with CUDA
  hidden. No package installation, GPU training or heavy profiling was needed.
- Skill frontmatter validation, relative Markdown links and diff formatting
  passed. A separate review checked the documented extension points against the
  actual collector, runtime-association and MFU implementations.
- **Independent agent integration passed:** an agent without the development
  conversation used the skill to connect two synthetic archived projects with
  JSONL and CSV logs. It produced a configuration, CSV adapter and local launcher
  reusing the supplied dashboard. Four focused checks exercised the actual HTTP
  API, canonical rank selection, distinct run identities, millisecond conversion,
  missing values, source timestamps, paused state and server cleanup. The expected
  API values were independently checked against the raw fixtures.
- These fixtures and generated adapters remain local test artifacts. No public
  demonstration is required. HTML delivery was checked, not browser rendering;
  real remote transport, distributed profiling and throughput improvements were
  not qualified by this archive-only exercise.
- Runtime code and dashboard HTML are unchanged from `d2e73dc`; this update
  changes the agent instructions and integration documentation. Publication
  also includes the previously local periodic MFU changes in that revision.

## First public release — 27 September 2026

Verification performed for the first public release. Test fixtures are not
published performance measurements.

- **48 core tests passed** with the Python standard-library environment. The
  optional PyTorch test was skipped there and run separately.
- **CPU PyTorch integration passed:** five updates with and without light hooks
  produced exactly equal model weights, optimizer state, RNG state and losses.
  This used PyTorch 2.7.0; it is a bounded correctness check, not a performance
  benchmark or a full-trajectory guarantee.
- **Two existing training logs adapted read-only:** latest progress fell within
  the source-read window, full chart history loaded, and diagnostic observations
  were preserved. Their paths/configuration remain under ignored `local/`.
- **Browser checks passed:** light phase timers, outside-step timing, diagnostics
  at step zero and latest-observation selection, metric/language switching,
  NVIDIA panels, time-window/power selection, and rolling raw dmon output.
  Desktop and 390-pixel mobile layouts had no horizontal overflow or JavaScript
  page errors in those checks. UI fixtures were explicitly synthetic.
- **Metric provenance checked:** valid sample counts, recent update-window
  boundaries, profiling flags, source timestamp versus file-time freshness,
  collection/publication timing, and configured sampling cadence. Missing
  values remain missing; the API exposes live units and source meanings.
- **Optional decision support checked:** disabled-by-default behavior, scoped
  measurement leads, immutable proposal content, matching configuration context,
  user approval/rejection/revocation, and agent-reported outcomes. Review writes
  reject a missing/incorrect page token, a foreign origin, and changed proposal
  content. These checks are not an authentication system against a malicious
  process with local filesystem access.
- **Browser review flow passed:** submit through the agent CLI, inspect the
  proposal, preserve a typed comment across refresh, approve and revoke. Desktop
  and mobile checks found no page errors or horizontal overflow. No training
  command was executed by the review flow.
- **Package build passed:** the wheel contains the dashboard HTML, recorder,
  worker and console entry points; local data and test artifacts are excluded.
- **Original dashboard source unchanged:** the copied source files retain their
  pre-work hashes. No training hook or profiler was installed into an active run.

The heavy replay worker was tested with a mock workflow adapter, real CSV-format
fixtures, and an actual bounded subprocess timeout. Success/failure releases
the adapter context, checkpoints are checked before use, and receipts prevent
automatic repeated attempts.

**Not qualified here:** an actual Nsight replay against a running training job,
physical GPU profiler overhead, multi-GPU/distributed equivalence, or recovery
for a particular scheduler after host failure. Those require the user's real
adapter and a bounded qualification window. Normal monitoring is independent
of that qualification.
