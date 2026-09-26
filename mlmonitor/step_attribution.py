"""Exclusive wall-time attribution from device events and explicit host ranges.

GPU activity counters are deliberately not inputs to this calculation.
"""
from collections import Counter
import math
import json

CATEGORIES = ('gpu_kernels', 'copy_h2d', 'copy_d2h', 'copy_d2d', 'memory_set',
              'launch_api', 'sync_api', 'copy_api', 'input_wait', 'cpu_prepare',
              'cpu_checks', 'optimizer', 'layout', 'forward_host', 'backward_host',
              'bookkeeping', 'host_other')
PRIORITY = {k: i for i, k in enumerate(CATEGORIES)}
FAMILIES = {
    'gpu': ('gpu_kernels',),
    'transfers': ('copy_h2d','copy_d2h','copy_d2d','memory_set','copy_api'),
    'input': ('input_wait','cpu_prepare','layout'),
    'host': ('launch_api','sync_api','cpu_checks','optimizer','forward_host','backward_host','bookkeeping','host_other'),
}


def partition(start, end, intervals):
    """Sweep a step once: device execution takes precedence over host overlap."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError('invalid step interval')
    boundaries = [(start, '', 0), (end, '', 0)]
    counts = Counter()
    totals = {k: 0. for k in CATEGORIES}
    for category, a, b in intervals:
        if category not in PRIORITY or not all(math.isfinite(x) for x in (a, b)):
            raise ValueError('invalid interval')
        a, b = max(start, a), min(end, b)
        if b > a:
            boundaries.extend(((a, category, 1), (b, category, -1)))
    previous = start
    for stamp, category, delta in sorted(boundaries):
        owner = min((k for k in CATEGORIES if counts[k]), key=PRIORITY.get, default='host_other')
        totals[owner] += stamp - previous
        if category:
            counts[category] += delta
        previous = stamp
    return totals


def classify(totals, valid=True):
    if not valid or sum(totals.values()) <= 0:
        return 'invalid_capture'
    ordered = sorted(totals, key=totals.get, reverse=True)
    first = ordered[0]
    return first if totals[first] / sum(totals.values()) >= .5 else 'mixed'


def summarize(start_ns, end_ns, host_ranges, events):
    """Kineto timestamps and host ranges share Unix ns. Ignore CUDA annotations."""
    intervals = [(k, (a-start_ns)/1e6, (b-start_ns)/1e6) for k, a, b in host_ranges]
    kernels = Counter()
    device_events = 0
    runtime_events = 0
    for event in events:
        if event.is_user_annotation():
            continue
        name = event.name()
        a, b = event.start_ns(), event.end_ns()
        if b <= start_ns or a >= end_ns:
            continue
        device = str(event.device_type()).split('.')[-1]
        category = None
        if device == 'CUDA':
            if name.startswith(('perf::','probe::')):
                continue
            device_events += 1
            low=name.lower().replace(' ','')
            if 'memset' in low: category='memory_set'
            elif 'memcpy' in low:
                category=('copy_h2d' if any(x in low for x in ('htod','h2d','hosttodevice')) else
                          'copy_d2h' if any(x in low for x in ('dtoh','d2h','devicetohost')) else 'copy_d2d')
            else:
                category='gpu_kernels'
                kernels[name] += (min(b,end_ns)-max(a,start_ns))/1e6
        elif device == 'CPU':
            low=name.lower()
            if 'launch' in low and low.startswith(('cuda','cu')):category='launch_api'
            elif 'synchronize' in low and low.startswith(('cuda','cu')):category='sync_api'
            elif ('memcpy' in low or 'memset' in low) and low.startswith(('cuda','cu')):category='copy_api'
            if category:runtime_events+=1
        if category:
            intervals.append((category, (a-start_ns)/1e6, (b-start_ns)/1e6))
    duration = (end_ns-start_ns)/1e6
    totals = partition(0., duration, intervals)
    valid = device_events > 0
    return {'valid': valid, 'classification': classify(totals, valid), 'step_ms': duration,
            'exclusive_ms': totals, 'fractions': {k: v/duration for k,v in totals.items()},
            'device_events': device_events, 'runtime_events': runtime_events,
            'kernel_top_ms': dict(kernels.most_common(12)),
            'family_ms': {k:sum(totals[x] for x in names) for k,names in FAMILIES.items()},
            'scope': 'optimizer step; GPU intervals first, uncovered host ranges next; excludes checkpoints and journal I/O'}


def load_capture(path, binding, active_pid, now, completed=False, current_step=None):
    """Read a dated, config-bound capture and label previous-process history."""
    if not path.exists():
        return {'status': 'not_profiled' if completed else 'waiting'}
    try:
        if path.stat().st_size > 100_000:
            raise ValueError('oversized capture')
        d = json.loads(path.read_text())
        if d['schema'] != 'instrumented-step-v2' or d['config_sha256'] != binding:
            return {'status': 'different_config'}
        if current_step is not None and d['update']>current_step:
            return {'status':'waiting'}
        age = now-d['captured_at']
        if age<0:return {'status':'invalid'}
        if (type(d.get('valid')) is not bool or type(d.get('update')) is not int or d['update']<0
                or type(d.get('step_ms')) not in (int,float) or not math.isfinite(d['step_ms']) or d['step_ms']<=0):
            raise ValueError('invalid capture metadata')
        values = d['exclusive_ms']
        if set(values) != set(CATEGORIES) or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('invalid measurements')
        if not math.isclose(sum(values.values()), d['step_ms'], rel_tol=1e-6, abs_tol=.001):
            raise ValueError('non-exclusive or incomplete timeline')
        d['classification'] = classify(values, d['valid'])
        return {'status': 'invalid' if not d['valid'] else 'perturbed' if d.get('timing_perturbed') else 'measured',
                'historical': completed or d.get('pid')!=active_pid, 'age': max(0,age), 'capture': d}
    except (OSError, ValueError, KeyError, TypeError):
        return {'status': 'invalid'}
