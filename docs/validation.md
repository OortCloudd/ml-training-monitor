# Local verification — 27 September 2026

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
