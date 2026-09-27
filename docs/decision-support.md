# Optional decision support

This mode adds measured investigation leads and a proposal/review workflow. It
is disabled by default and works without an LLM service or external account.
The dashboard never edits training code, starts a profiler, or executes an
approved intervention. Your coding agent remains the executor, within the
specific scope the user approves.

## Opt in

After the user chooses this mode, add to their local configuration and restart
the dashboard:

```json
"decision_support": {
  "enabled": true,
  "directory": "local/proposals"
}
```

The directory stores local proposal and review records. Keep it out of source
control. Disabling the setting restores ordinary monitoring without the review
panel. Existing records are retained, not silently deleted.

## What the built-in suggestions mean

The engine uses existing measurements: large recorded groups in a valid,
configuration-matched step capture; aligned ready-batch wait timers; and costly
outside-step operations. It ranks a few leads, retaining their source, date,
window and numerical observation. Without those measurements it suggests a
bounded observation of the step rather than guessing from GPU activity.

The 10% screening threshold for aligned wait/operation timers is a prioritization
heuristic, not a physical bottleneck definition. Profile groups use the existing
GPU-first interval accounting. Neither heuristic proves a cause or predicts a
speedup. An agent may propose a different intervention based on code inspection;
it is not restricted to the built-in leads.

Read the leads and proposals through `GET /api/decisions`, or the agent CLI:

```bash
python3 -m mlmonitor.proposals --config monitor.local.json suggest
python3 -m mlmonitor.proposals --config monitor.local.json list
```

Use `--url http://your-configured-host:port/api/metrics` before the subcommand
when the dashboard uses a different address. No proposal is generated or
approved just by refreshing the monitoring page.

## Prepare a concrete intervention

The agent inspects the actual project and prepares a small plan JSON containing:

| Field | What the user should be able to judge |
| --- | --- |
| `title` | The specific proposed intervention |
| `target_revision` | Actual revision/dirty-patch identity or equivalent code identity |
| `change` | Exact parameter changes, affected files or a bounded patch; include any profiling interruption |
| `rationale` | Why these observations/code findings motivate the change |
| `expected_effect` | Expected effect, clearly distinguished from a measured gain |
| `validation` | How useful throughput and the affected behavior will be checked |
| `risks` | Resource cost, behavior tradeoffs and possible failure modes |
| `rollback` | How to retain/restore the previous implementation and training state |

Do the authorized, reversible preparation needed to make the proposal concrete.
Do not apply it to active training before the user's decision. A vague request
to “optimize the GPU” is not a useful proposal even if it passes JSON validation.

```bash
python3 -m mlmonitor.proposals --config monitor.local.json submit \
  --run my-run --plan local/prefetch-plan.json
```

Optionally select a particular lead using `--opportunity <id>`. The submission
attaches the observed run/configuration context and measurements, computes an
immutable content digest and starts in `pending` state. Updating a plan means
submitting a new proposal, not modifying one the user already reviewed.

## User review and agent follow-through

The dashboard shows the full intervention, evidence, expected effect, validation,
tradeoffs and rollback. The user can **approve this scope**, **reject**, or
**revoke approval**, with an optional comment. Approval records the exact content
digest. A changed known run/configuration requires an updated proposal.

The agent reads the latest record immediately before execution. It must check
`status == "approved"`, `context_matches == true`, the exact `content_sha256`,
and the stated code revision against the actual project. The dashboard can
compare registered configuration identity; it does not inspect an arbitrary
training repository to verify its revision. A missing configuration fingerprint
is shown explicitly in the review panel, so the user can judge the stated scope.

Enabling the mode is not blanket approval. Agents must not call the review
endpoint, impersonate a user decision, or mark themselves approved. Approved
work must remain within the reviewed scope. Revocation is not a kill command:
if work has already started, the user and agent must handle that explicitly.

After execution, the agent can record the actual result:

```bash
python3 -m mlmonitor.proposals --config monitor.local.json outcome \
  --id <proposal-id> --content-sha256 <approved-content-digest> \
  --status completed --summary-file local/result.txt
```

Other outcome statuses are `failed` and `reverted`. Describe measurements and
behavior checks honestly. The UI labels these as **agent-reported outcomes**,
not automatically verified speedups. Approval history remains attached.

## Deployment boundary

This is a single-user local review workflow, not an authentication system or a
sandbox against a malicious process with filesystem access. Use loopback/SSH
tunnelling or the user's trusted private access controls. Review writes require
a page token, matching browser origin and exact proposal digest. The token is
not included in the metrics or proposal JSON APIs. Reads remain available
without a write token.

For remote review, bind to the explicit private address and access that address,
or use an SSH tunnel to loopback. A wildcard listener is not permission to review
through arbitrary hostnames. An authenticated reverse proxy needs corresponding
origin integration before review writes will work; this version deliberately
does not trust forwarded headers automatically.
