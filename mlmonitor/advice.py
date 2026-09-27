"""Optional, local decision support: measured leads and user-reviewed proposals.

This module never executes training commands. Suggestions are starting points
for project-specific work, not certified causal diagnoses or promised gains.
"""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import re
import time
import uuid

from .storage import atomic, digest, finite, read
from .step_attribution import CATEGORIES

PLAN_FIELDS = ('title', 'target_revision', 'change', 'rationale', 'expected_effect',
               'validation', 'risks', 'rollback')
ID_PATTERN = re.compile(r'[a-f0-9]{32}')


def opportunities(snapshot):
    """Rank a few measured leads; keep all scopes, dates and counts attached."""
    result = []
    for run in snapshot.get('runs', []):
        leads = []

        def lead(key, title, measured, experiment, evidence, priority=0):
            leads.append({'id': key, 'run_id': run['id'], 'title': title,
                          'measured': measured, 'experiment': experiment,
                          'evidence': evidence, '_priority': priority})

        profile = run.get('bottleneck', {}).get('capture')
        if isinstance(profile, dict):
            values, duration = profile.get('exclusive_ms', {}), finite(profile.get('step_ms'))
            good = (profile.get('valid') is True and bool(run.get('config_sha256')) and
                    profile.get('config_sha256') == run.get('config_sha256') and
                    finite(profile.get('captured_at')) is not None and
                    profile['captured_at'] <= (finite(snapshot.get('collection',{}).get('published_at')) or
                                               finite(snapshot.get('timestamp')) or time.time()) and
                    finite(profile.get('update')) is not None and
                    0 <= profile['update'] <= (finite(run.get('step')) or 0) and
                    set(values) == set(CATEGORIES) and duration is not None and duration > 0 and
                    all(finite(v) is not None and v >= 0 for v in values.values()) and
                    math.isclose(sum(values.values()), duration, rel_tol=1e-6, abs_tol=.001))
            if good:
                groups = {
                    'input': (('input_wait', 'cpu_prepare', 'layout'), 'Input preparation and waiting',
                              'Inspect the producer/queue/assembly path and benchmark one equivalent prefetch or preparation change.'),
                    'transfers': (('copy_h2d', 'copy_d2h', 'copy_d2d', 'memory_set', 'copy_api'), 'Data movement',
                                  'Locate the expensive copy; benchmark removing it or overlapping it while preserving values and buffer lifetimes.'),
                    'host': (('launch_api', 'sync_api', 'cpu_checks', 'optimizer', 'forward_host', 'backward_host', 'bookkeeping', 'host_other'), 'Host work and dispatch',
                             'Inspect the dominant host ranges and call sites; benchmark one targeted dispatch, synchronization or CPU-work change.'),
                    'kernels': (('gpu_kernels',), 'GPU kernel execution',
                                'Inspect the hottest kernel family and representative shapes; compare a compatible kernel/fusion/layout implementation. Use targeted counters when helpful.'),
                }
                for key, (names, title, experiment) in groups.items():
                    ms = sum(values[name] for name in names)
                    if ms <= 0:
                        continue
                    lead('profile_' + key, title,
                         f'{ms:.3f} ms of a {duration:.3f} ms captured update ({100*ms/duration:.1f}%).', experiment,
                         {'source': 'framework_capture', 'captured_at': profile.get('captured_at'),
                          'update': profile.get('update'), 'milliseconds': ms, 'step_ms': duration,
                          'scope': 'Dated instrumented step, GPU-first interval accounting. This locates time; it does not prove a cause.'},
                         ms / duration)

        light = run.get('light_profile') or {}
        wait = light.get('phases', {}).get('input_wait', {})
        samples = finite(wait.get('steps')) or finite(light.get('steps'))
        wait_seconds, step_seconds = finite(wait.get('seconds')), finite(run.get('seconds'))
        window = run.get('telemetry_window', {})
        aligned = (light.get('first_update') == window.get('first_update') and
                   light.get('last_update') == window.get('last_update') and bool(window.get('records')))
        if not leads and samples and samples > 0 and wait_seconds is not None and wait_seconds > 0:
            mean = wait_seconds / samples
            # Screening priority only, not a physical bottleneck threshold.
            if aligned and step_seconds and step_seconds > 0 and mean / step_seconds >= .1:
                lead('input_wait', 'Ready-batch waiting',
                     f'{mean:.4f} seconds of inclusive input waiting per observed update; {samples:g} observations.',
                     'Inspect the producer and ready-batch queue; benchmark one input-delivery change against the unchanged prepared tensors and source order.',
                     {'source': 'host_phase_timers', 'captured_at': light.get('captured_at'),
                      'first_update': light.get('first_update'), 'last_update': light.get('last_update'),
                      'samples': samples, 'mean_wait_seconds': mean,
                      'scope': 'Inclusive host timer; overlap must be accounted for when interpreting a speed change.'}, mean / step_seconds)

        operations = run.get('operations') or {}
        op_window = finite(operations.get('window_seconds'))
        for name, value in operations.get('operations', {}).items():
            seconds = finite(value.get('seconds'))
            if seconds and op_window and op_window > 0 and seconds / op_window >= .1:
                lead('operation_' + name, 'Outside-step work: ' + name,
                     f'{seconds:.3f} seconds recorded over a {op_window:.3f}-second window.',
                     'Inspect this operation and test equivalent execution/serialization improvements while preserving its required frequency and outputs.',
                     {'source': 'operation_timers', 'captured_at': operations.get('captured_at'),
                      'seconds': seconds, 'window_seconds': op_window,
                      'scope': 'Inclusive operation timer; nested operations may overlap.'}, seconds / op_window)

        if not leads:
            lead('observe_step', 'Locate the expensive part of an update',
                 f"Latest recorded update: {run.get('step', 0)}; recent timing samples: {window.get('valid_samples', {}).get('seconds', 0)}.",
                 'Inspect the existing timing/code boundaries and add light phase timers or one bounded representative capture, then target the largest avoidable cost.',
                 {'source': 'training_telemetry', 'snapshot_at': snapshot.get('timestamp'),
                  'telemetry_window': window,
                  'scope': 'Device activity alone does not select a compute, bandwidth or host optimization.'})
        for item in sorted(leads, key=lambda x: x['_priority'], reverse=True)[:3]:
            item.pop('_priority')
            item['note'] = 'Investigation lead, not a measured speedup. The agent must turn it into a concrete proposal for user review.'
            result.append(item)
    return result


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) - set(PLAN_FIELDS):
        raise ValueError('proposal contains unknown fields')
    for key in PLAN_FIELDS:
        value = plan.get(key)
        limit = 200 if key == 'title' else 8000
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f'{key} must be a nonempty string of at most {limit} characters')
    return {key: plan[key].strip() for key in PLAN_FIELDS}


def proposal_context(snapshot, run_id):
    run = next((r for r in snapshot.get('runs', []) if r.get('id') == run_id), None)
    if run is None:
        raise ValueError('run is not registered in the current snapshot')
    return {'run_id': run_id, 'run_name': run.get('name', run_id),
            'config_sha256': run.get('config_sha256'), 'update': run.get('step'),
            'snapshot_at': snapshot.get('timestamp'), 'freshness': run.get('freshness'),
            'telemetry_window': run.get('telemetry_window')}


class ProposalStore:
    """Immutable proposal content, with reviews stored against its exact digest."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.directory / '.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def path(self, ident):
        if not isinstance(ident, str) or not ID_PATTERN.fullmatch(ident):
            raise ValueError('invalid proposal ID')
        return self.directory / (ident + '.json')

    def get(self, ident):
        record = read(self.path(ident), limit=250_000)
        if (record.get('schema') != 'ml-monitor-proposal-v1' or record.get('id') != ident or
                not isinstance(record.get('content'), dict) or
                digest(record['content']) != record.get('content_sha256')):
            raise ValueError('proposal is absent, corrupt or its content was changed')
        validate_plan(record['content']['plan'])
        if (finite(record.get('created_at')) is None or
                record.get('status') not in ('pending','approved','rejected','revoked','completed','failed','reverted') or
                not isinstance(record.get('reviews'),list) or
                not isinstance(record['content'].get('context'),dict) or
                not isinstance(record['content'].get('evidence'),list)):
            raise ValueError('invalid proposal record')
        reviews=record['reviews']
        if any(not isinstance(r,dict) or r.get('decision') not in ('approved','rejected','revoked') or
               r.get('content_sha256')!=record['content_sha256'] for r in reviews):
            raise ValueError('invalid review history')
        expected={'approved':'approved','rejected':'rejected','revoked':'revoked',
                  'completed':'approved','failed':'approved','reverted':'approved'}.get(record['status'])
        if (record['status']=='pending' and reviews) or (expected and (not reviews or reviews[-1]['decision']!=expected)):
            raise ValueError('status does not match the recorded review')
        return record

    def submit(self, snapshot, run_id, plan, opportunity_id=None):
        context = proposal_context(snapshot, run_id)
        evidence = [p for p in opportunities(snapshot) if p['run_id'] == run_id]
        if opportunity_id:
            evidence = [p for p in evidence if p['id'] == opportunity_id]
            if not evidence:
                raise ValueError('opportunity is not present in this snapshot')
        content = {'plan': validate_plan(plan), 'context': context, 'evidence': evidence}
        record = {'schema': 'ml-monitor-proposal-v1', 'id': uuid.uuid4().hex,
                  'created_at': time.time(), 'content': content, 'content_sha256': digest(content),
                  'status': 'pending', 'reviews': [], 'outcome': None}
        with self.locked():
            atomic(self.path(record['id']), record)
        return record

    @staticmethod
    def context_matches(record, snapshot):
        context = record['content']['context']
        run = next((r for r in snapshot.get('runs', []) if r.get('id') == context['run_id']), None)
        return run is not None and run.get('config_sha256') == context.get('config_sha256')

    def listing(self, snapshot):
        records, errors = [], 0
        for path in self.directory.glob('*.json'):
            try:
                record = self.get(path.stem)
                record['context_matches'] = self.context_matches(record, snapshot)
                records.append(record)
            except (OSError, ValueError, KeyError, TypeError):
                errors += 1
        records.sort(key=lambda r: r['created_at'], reverse=True)
        return {'proposals': records[:50], 'older_proposals': max(0, len(records)-50), 'invalid_records': errors}

    def review(self, ident, content_sha256, decision, comment, snapshot):
        if decision not in ('approved', 'rejected', 'revoked'):
            raise ValueError('review must approve, reject or revoke')
        if not isinstance(comment, str) or len(comment) > 2000:
            raise ValueError('review comment is too long')
        with self.locked():
            record = self.get(ident)
            if content_sha256 != record['content_sha256']:
                raise ValueError('proposal differs from the reviewed content')
            allowed = {'pending': ('approved', 'rejected'), 'approved': ('revoked',)}
            if decision not in allowed.get(record['status'], ()):
                raise ValueError('proposal is no longer in a reviewable state')
            if decision == 'approved' and not self.context_matches(record, snapshot):
                raise ValueError('run/configuration changed; submit a new proposal')
            record['reviews'].append({'decision': decision, 'at': time.time(), 'comment': comment,
                                      'content_sha256': content_sha256, 'channel': 'dashboard_review'})
            record['status'] = decision
            atomic(self.path(ident), record)
        return record

    def record_outcome(self, ident, content_sha256, status, summary):
        if status not in ('completed', 'failed', 'reverted') or not isinstance(summary, str) or not 1 <= len(summary) <= 8000:
            raise ValueError('outcome needs completed/failed/reverted status and a short measured-result summary')
        with self.locked():
            record = self.get(ident)
            if record['status'] != 'approved' or content_sha256 != record['content_sha256']:
                raise ValueError('this exact proposal has not been approved, or approval was revoked')
            record['outcome'] = {'status': status, 'summary': summary, 'at': time.time(), 'source': 'agent_report'}
            record['status'] = status
            atomic(self.path(ident), record)
        return record
