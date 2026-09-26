"""Explicit training hooks, derived from the working dashboard's recorder.

Light monitoring does not import torch, synchronize devices, or modify training
objects. Optional profiling is selected separately. Observer failures never
replace an exception from the training loop.
"""
from collections import deque
from contextlib import contextmanager
import os
from pathlib import Path
import re
import statistics
import threading
import time

from .storage import append, atomic, digest, finite, read
from .step_attribution import CATEGORIES, summarize


def next_capture(update, every=None, steps=()):
    candidates = [s for s in steps if s > update]
    if every:
        candidates.append((update // every + 1) * every)
    return min(candidates, default=None)


def due(update, every=None, steps=()):
    return update > 0 and (update in steps or bool(every and update % every == 0))


class TorchCapture:
    """Opt-in, single-device PyTorch capture; synchronization occurs only here."""
    def __init__(self, device='cuda:0'):
        self.device = device

    def start(self):
        import torch
        self.torch = torch
        torch.cuda.synchronize(self.device)
        self.profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                              record_shapes=False, profile_memory=False, with_stack=False)
        self.profile.__enter__()

    def boundary(self):
        self.torch.cuda.synchronize(self.device)

    def finish(self):
        self.profile.__exit__(None, None, None)
        index=self.torch.device(self.device).index
        if index is None:index=self.torch.cuda.current_device()
        return [event for event in self.profile.profiler.kineto_results.events()
                if str(event.device_type()).split('.')[-1]!='CUDA' or event.device_index()==index]


class Observation:
    def __init__(self):
        self.values = {}

    def record(self, *, loss=None, gradient_norm=None, step_seconds=None):
        # The caller supplies existing scalars; never call .item() on a tensor.
        self.values = {'loss': finite(loss), 'gradient': finite(gradient_norm), 'seconds': finite(step_seconds)}


class Monitor:
    def __init__(self, directory, *, run_id, config, target_updates=None, details=None,
                 profile_every=None, profile_steps=(), capture_factory=None, device='cuda:0',
                 heavy_every=None, heavy_steps=(), start_step=0, publish_seconds=5):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', run_id):
            raise ValueError('run_id must contain letters, digits, underscores or hyphens')
        for value in (profile_every, heavy_every, target_updates):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError('cadences and budgets must be positive integers')
        for steps in (profile_steps, heavy_steps):
            if any(type(s) is not int or s < 1 for s in steps):
                raise ValueError('milestones must be positive integers')
        if type(start_step) is not int or start_step < 0:
            raise ValueError('start_step must be a nonnegative integer')
        self.directory = Path(directory)
        self.run_id, self.config_sha = run_id, digest(config)
        self.target = target_updates
        if not isinstance(publish_seconds,(int,float)) or publish_seconds<0:
            raise ValueError('publish_seconds must be nonnegative')
        self.publish_seconds,self.last_publish=publish_seconds,-float('inf')
        self.every, self.steps = profile_every, tuple(profile_steps)
        self.heavy_every, self.heavy_steps = heavy_every, tuple(heavy_steps)
        self.capture_factory = capture_factory or (lambda: TorchCapture(device))
        self.last_update = start_step
        self.profile_disabled = self.disabled = False
        self.error = None
        self.active = None
        self.host_ranges = []
        self.phases = {}
        self.operations = {}
        self.operation_start = time.monotonic()
        self.recent = deque(maxlen=100)
        self.baselines = deque(maxlen=30)
        self.thread = threading.get_ident()
        bindings = read(self.directory / 'run_bindings.json')
        if bindings and (bindings.get('config_sha256') != self.config_sha or bindings.get('run_id') != run_id):
            raise ValueError('directory belongs to a different run/configuration; choose a new directory')
        self.state = {'status': 'RUNNING', 'target_updates': target_updates, 'last_update': start_step,
                      'durable_checkpoint_update': None}
        old = read(self.directory / 'state.json')
        if old.get('config_sha256') == self.config_sha:
            self.state['durable_checkpoint_update'] = old.get('durable_checkpoint_update')
        self._write('run_bindings.json', {'run_id': run_id, 'config_sha256': self.config_sha,
            'details': details or {}, 'profile_every': profile_every, 'profile_steps': list(profile_steps),
            'heavy_every': heavy_every, 'heavy_steps': list(heavy_steps)})
        self._publish_state()

    def _error(self, exc):
        self.error = type(exc).__name__
        print('ML monitor: observation failed (' + self.error + ')', flush=True)

    def _write(self, name, value):
        try:
            atomic(self.directory / name, value)
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def _append(self, name, value):
        try:
            append(self.directory / name, value)
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def _publish_state(self):
        self._write('state.json', {**self.state, 'config_sha256': self.config_sha, 'observer_error': self.error,
                                  'profile_disabled': self.profile_disabled,
                                  'next_capture_update': next_capture(self.last_update, self.every, self.steps)})

    @contextmanager
    def phase(self, name):
        if name not in CATEGORIES or name.startswith('gpu_') or name.startswith('copy_') or name == 'memory_set':
            raise ValueError('use a named host phase from step_attribution.CATEGORIES')
        if self.active is None or threading.get_ident() != self.thread:
            yield
            return
        start, wall = time.perf_counter_ns(), time.time_ns()
        try:
            yield
        finally:
            elapsed = (time.perf_counter_ns() - start) / 1e9
            entry = self.phases.setdefault(name, {'seconds': 0., 'calls': 0})
            entry['seconds'] += elapsed
            entry['calls'] += 1
            if self.active.get('capture') is not None:
                self.host_ranges.append((name, wall, time.time_ns()))

    @contextmanager
    def operation(self, name):
        """Time work outside optimizer steps: checkpoint, evaluation, journal, etc."""
        if not re.fullmatch('[a-zA-Z0-9_-]{1,64}', name):
            raise ValueError('invalid operation label')
        start = time.monotonic()
        try:
            yield
        finally:
            entry = self.operations.setdefault(name, {'seconds': 0., 'calls': 0})
            entry['seconds'] += time.monotonic() - start
            entry['calls'] += 1
            self._publish_operations()

    def _publish_operations(self):
        self._write('operations.json', {'config_sha256': self.config_sha, 'run_id': self.run_id,
            'pid': os.getpid(), 'captured_at': time.time(), 'window_seconds': time.monotonic() - self.operation_start,
            'operations': self.operations})

    @contextmanager
    def step(self, update):
        if type(update) is not int or update <= self.last_update or self.active is not None:
            raise ValueError('step needs the next increasing completed-update index; steps cannot nest')
        observation = Observation()
        capture = None
        capture_total_start = time.monotonic()
        milestone_path = self.directory / 'profiles' / f'{update:012d}.json'
        if not self.profile_disabled and due(update, self.every, self.steps) and not milestone_path.exists():
            candidate = None
            try:
                candidate = self.capture_factory()
                candidate.start()
                capture = candidate
            except Exception as exc:
                self._error(exc)
                self.profile_disabled = True
        self.host_ranges, self.phases = [], {}
        self.active = {'capture': capture}
        wall_start, mono_start = time.time_ns(), time.perf_counter_ns()
        success = False
        try:
            yield observation
            success = True
        finally:
            if capture is not None:
                boundary_valid=True
                try:
                    capture.boundary()
                except Exception as exc:
                    self._error(exc)
                    self.profile_disabled = True
                    boundary_valid=False
            end_ns, elapsed = time.time_ns(), (time.perf_counter_ns() - mono_start) / 1e9
            self.active = None
            events = None
            if capture is not None:
                try:
                    events = capture.finish()
                except Exception as exc:
                    self._error(exc)
                    self.profile_disabled = True
            if success:
                self.last_update = update
                supplied = observation.values
                seconds = supplied.get('seconds')
                seconds = elapsed if seconds is None else seconds
                row = {'update': update, 'timestamp': time.time(), 'loss_components': {'loss': supplied.get('loss')},
                       'encoder_gradient_norm': supplied.get('gradient'), 'update_seconds': seconds,
                       'host_scope_seconds': elapsed, 'profiled': capture is not None}
                for phase, key in [('input_wait', 'prepared_input_wait_seconds'), ('cpu_prepare', 'input_prepare_seconds')]:
                    row[key] = self.phases.get(phase, {}).get('seconds')
                self._append('training_telemetry.jsonl', row)
                self.recent.append((update, elapsed, dict(self.phases)))
                if events is not None:
                    try:
                        result = summarize(wall_start, end_ns, self.host_ranges, events)
                        if not boundary_valid or abs((end_ns - wall_start) / 1e9 - elapsed) > .005:
                            result['valid'] = False
                        result.update(schema='instrumented-step-v2', run_id=self.run_id, config_sha256=self.config_sha,
                            update=update, pid=os.getpid(), captured_at=time.time(),
                            baseline_step_ms=statistics.median(self.baselines) if self.baselines else None,
                            baseline_steps=len(self.baselines), capture_total_ms=(time.monotonic() - capture_total_start) * 1000,
                            next_capture_update=next_capture(update, self.every, self.steps), timing_perturbed=True)
                        result['scope'] = 'Opt-in synchronized optimizer step; GPU intervals first, then uncovered host ranges; excludes logging/checkpoint/evaluation'
                        self._write(str(milestone_path.relative_to(self.directory)), result)
                        self._write('profile.json', result)
                        self._append('profile_history.jsonl', result)
                    except Exception as exc:
                        self._error(exc)
                        self.profile_disabled = True
                if capture is None:
                    self.baselines.append(seconds * 1000)
                self.state['last_update'] = update
                if time.monotonic()-self.last_publish>=self.publish_seconds or capture is not None:
                    self.flush()
            else:
                self.state['status'] = 'FAILED'
                self._publish_state()

    def _publish_phases(self):
        totals = {}
        for _, _, phases in self.recent:
            for name, item in phases.items():
                entry = totals.setdefault(name, {'seconds': 0., 'calls': 0, 'steps': 0})
                entry['seconds'] += item['seconds']
                entry['calls'] += item['calls']
                entry['steps'] += 1
        self._write('phase_timings.json', {'run_id': self.run_id, 'config_sha256': self.config_sha,
            'captured_at': time.time(), 'first_update': self.recent[0][0], 'last_update': self.recent[-1][0],
            'steps': len(self.recent), 'step_seconds': sum(r[1] for r in self.recent), 'phases': totals,
            'scope': 'Inclusive host wall timers; asynchronous device execution may overlap these intervals'})

    def diagnostic(self, update, metrics, *, healthy=None):
        if type(update) is not int or update < 0:
            raise ValueError('diagnostic update must be nonnegative')
        values = {str(k): finite(v) for k, v in metrics.items()}
        self._append('health_events.jsonl', {'update': update, 'timestamp': time.time(),
                                           'healthy': healthy if type(healthy) is bool else None, 'summary': values})

    def checkpoint(self, update, path, *, terminal=False):
        """Called AFTER the user's durable save. Never saves or loads weights."""
        try:
            if type(update) is not int or update < 0 or update > self.last_update:
                raise ValueError('checkpoint update is outside observed progress')
            path = Path(path).resolve()
            stat = path.stat()
            if not path.is_file():
                raise ValueError('checkpoint must be a file')
            self.state['durable_checkpoint_update'] = update
            self._publish_state()
            if not due(update, self.heavy_every, self.heavy_steps):
                return
            key = f'{self.run_id}-{self.config_sha[:16]}-{update:012d}'
            if (self.directory / 'requests' / (key + '.json')).exists() or (self.directory / 'receipts' / (key + '.json')).exists():
                return
            self._write('requests/' + key + '.json', {'schema': 'ml-monitor-capture-request-v1', 'key': key,
                'run_id': self.run_id, 'config_sha256': self.config_sha, 'update': update, 'terminal': bool(terminal),
                'checkpoint': str(path), 'checkpoint_size': stat.st_size, 'checkpoint_mtime_ns': stat.st_mtime_ns,
                'created_at': time.time()})
        except Exception as exc:
            self._error(exc)

    def close(self, status='COMPLETE'):
        if status not in ('COMPLETE', 'PAUSED', 'FAILED'):
            raise ValueError('status must be COMPLETE, PAUSED or FAILED')
        self.state['status'] = status
        self.flush()

    def flush(self):
        if self.recent:self._publish_phases()
        self._publish_operations()
        self._publish_state()
        self.last_publish=time.monotonic()
