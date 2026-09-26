"""Deployment settings and adapters for existing scalar logs."""
import json
from pathlib import Path
import re

FIELDS = {'step': 'update', 'loss': 'loss_components.loss', 'seconds': 'update_seconds',
          'gradient': 'encoder_gradient_norm', 'timestamp': 'timestamp',
          'profiled': 'profiled',
          'source_read_seconds': 'source_read_seconds', 'input_prepare_seconds': 'input_prepare_seconds',
          'prepared_input_wait_seconds': 'prepared_input_wait_seconds'}
FILES = {'telemetry': 'training_telemetry.jsonl', 'state': 'state.json',
         'health': 'health_events.jsonl', 'bindings': 'run_bindings.json',
         'profile': 'profile.json', 'light_profile': 'phase_timings.json',
         'operations': 'operations.json', 'profiles': 'profile_history.jsonl', 'captures': 'nsight'}


def load_config(path):
    path = Path(path).expanduser().resolve()
    def invalid(value):
        raise ValueError('non-finite config value: ' + value)
    raw = json.loads(path.read_text(), parse_constant=invalid)
    if not isinstance(raw, dict) or set(raw) - {'title', 'interval', 'gpu_indices', 'gpu_enabled', 'disks', 'runs'}:
        raise ValueError('unknown dashboard settings')
    result = {'title': raw.get('title', 'ML Training Monitor'), 'interval': raw.get('interval', 2),
              'gpu_enabled': raw.get('gpu_enabled', True), 'gpu_indices': raw.get('gpu_indices'), 'runs': []}
    if not isinstance(result['title'], str) or len(result['title']) > 160:
        raise ValueError('title must be a short string')
    if type(result['interval']) is not int or not 1 <= result['interval'] <= 60:
        raise ValueError('interval must be 1–60 seconds')
    if type(result['gpu_enabled']) is not bool:
        raise ValueError('gpu_enabled must be boolean')
    indices = result['gpu_indices']
    if indices is not None and (not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices)):
        raise ValueError('gpu_indices must be null or a list of physical GPU indices')
    def resolve(value, root=path.parent):
        if not isinstance(value, str) or not value:
            raise ValueError('paths must be nonempty strings')
        p = Path(value).expanduser()
        return (root / p).resolve()
    result['disks'] = []
    for disk in raw.get('disks', []):
        if not isinstance(disk, dict) or set(disk) != {'label', 'path'} or not isinstance(disk['label'], str):
            raise ValueError('disks require label and path')
        result['disks'].append({'label': disk['label'], 'path': resolve(disk['path'])})
    ids = set()
    for r in raw.get('runs', []):
        allowed = {'id', 'name', 'directory', 'files', 'fields', 'target_updates', 'step_unit', 'diagnostics',
                   'diagnostic_fields', 'details', 'config_sha256', 'capture_every', 'capture_steps'}
        if not isinstance(r, dict) or set(r) - allowed:
            raise ValueError('unknown run settings')
        key = r.get('id')
        if not isinstance(key, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', key) or key in ids:
            raise ValueError('run IDs must be unique and contain letters, digits, underscores or hyphens')
        ids.add(key)
        folder = resolve(r['directory'])
        files = r.get('files', {})
        fields = r.get('fields', {})
        if files.get('telemetry',FILES['telemetry']) is None:
            raise ValueError('telemetry cannot be null; omit unstarted runs or register their future log path')
        if set(files) - set(FILES) or set(fields) - set(FIELDS):
            raise ValueError('unknown file or metric mapping')
        fields = {**FIELDS, **fields}
        if fields['step'] is None or any(v is not None and (not isinstance(v, str) or not v) for v in fields.values()):
            raise ValueError('metric mappings require dot-separated names; optional mappings can be null')
        details = r.get('details', {})
        if not isinstance(details, dict) or any(not isinstance(k, str) or not isinstance(v, (str, int, float, bool)) for k, v in details.items()):
            raise ValueError('details must contain labels and scalar values for display')
        diagnostics = r.get('diagnostics', [])
        diagnostic_keys = set()
        for d in diagnostics:
            if (not isinstance(d, dict) or set(d) - {'key', 'label', 'unit', 'reference', 'help'} or
                not isinstance(d.get('key'), str) or not re.fullmatch(r'[a-zA-Z0-9_]{1,64}', d['key']) or
                not isinstance(d.get('label'), str) or d['key'] in diagnostic_keys):
                raise ValueError('diagnostics need unique keys and display labels')
            diagnostic_keys.add(d['key'])
            if d.get('reference') is not None and type(d['reference']) not in (int, float):
                raise ValueError('reference must be numeric or omitted')
        df = {'step': 'update', 'healthy': 'healthy', **{d['key']: 'summary.' + d['key'] for d in diagnostics}, **r.get('diagnostic_fields', {})}
        if set(df) - ({'step', 'healthy'} | diagnostic_keys) or any(v is not None and not isinstance(v, str) for v in df.values()):
            raise ValueError('unknown diagnostic mapping')
        for k in ('target_updates', 'capture_every'):
            if r.get(k) is not None and (type(r[k]) is not int or r[k] <= 0):
                raise ValueError(k + ' must be a positive integer')
        steps = r.get('capture_steps', [])
        if not isinstance(steps, list) or any(type(s) is not int or s < 1 for s in steps):
            raise ValueError('capture_steps must be positive integers')
        config_sha = r.get('config_sha256')
        if config_sha is not None and (not isinstance(config_sha, str) or not re.fullmatch('[a-f0-9]{64}', config_sha)):
            raise ValueError('config_sha256 must be a lowercase SHA-256 digest')
        result['runs'].append({**r, 'id': key, 'name': r.get('name', key), 'directory': folder,
            'files': {k: resolve(v, folder) if v is not None else None for k, v in {**FILES, **files}.items()},
            'fields': fields, 'diagnostic_fields': df, 'diagnostics': diagnostics, 'details': details,
            'step_unit': r.get('step_unit', 'optimizer updates')})
    return result
