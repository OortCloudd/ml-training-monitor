"""Explicit useful-model FLOP accounting; no model execution or GPU imports."""
import math
import re

from .storage import finite


def positive(value):
    return finite(value) is not None and value > 0


def validate_contract(value):
    if value is None:
        return None
    required = {'config_sha256', 'gpu_model', 'gpu_count', 'precision', 'sparsity',
                'peak_tflops_per_gpu', 'peak_source', 'flop_source', 'timing_scope', 'mode'}
    allowed = required | {'flops_per_update', 'flops_per_update_bounds',
                          'dataset_manifest_sha256', 'assumptions', 'gpu_uuids', 'estimator_sha256'}
    if not isinstance(value, dict) or required - value.keys() or value.keys() - allowed:
        raise ValueError('mfu requires explicit FLOP, timing, configuration and hardware provenance')
    for key in ('config_sha256', 'dataset_manifest_sha256', 'estimator_sha256'):
        if key in value and (not isinstance(value[key], str) or not re.fullmatch('[a-f0-9]{64}', value[key])):
            raise ValueError('mfu ' + key + ' must be a SHA-256 digest')
    for key in ('gpu_model', 'precision', 'peak_source', 'flop_source', 'timing_scope'):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 2000:
            raise ValueError('mfu ' + key + ' must be a nonempty description')
    if value['sparsity'] not in ('dense', 'structured_sparse'):
        raise ValueError('mfu sparsity must match the workload and hardware peak')
    if type(value['gpu_count']) is not int or value['gpu_count'] < 1 or not positive(value['peak_tflops_per_gpu']):
        raise ValueError('mfu requires positive GPU count and peak')
    if not positive(value['gpu_count'] * value['peak_tflops_per_gpu'] * 1e12):
        raise ValueError('mfu hardware peak overflows')
    if value['mode'] not in ('per_step', 'constant', 'bounds', 'capture'):
        raise ValueError('mfu mode must be per_step, constant, bounds or capture')
    point = value.get('flops_per_update')
    bounds = value.get('flops_per_update_bounds')
    if value['mode'] == 'constant' and not positive(point):
        raise ValueError('constant mfu requires positive flops_per_update')
    if value['mode'] != 'constant' and point is not None:
        raise ValueError('mfu point count belongs only to constant mode')
    if value['mode'] == 'bounds' or bounds is not None:
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                not all(positive(v) for v in bounds) or bounds[0] > bounds[1]):
            raise ValueError('mfu requires ordered positive FLOP bounds')
        if point is not None and not bounds[0] <= point <= bounds[1]:
            raise ValueError('mfu point falls outside its FLOP bounds')
    if value['mode'] == 'per_step' and bounds is not None:
        raise ValueError('per_step mfu uses logged counts, not constant bounds')
    assumptions = value.get('assumptions', [])
    if not isinstance(assumptions, list) or any(not isinstance(s, str) or len(s) > 2000 for s in assumptions):
        raise ValueError('mfu assumptions must be a list of descriptions')
    if value['mode']=='capture':
        if value['precision'].lower() not in ('bf16','fp16','fp32','fp64'):
            raise ValueError('capture requires a supported single arithmetic precision')
        uuids=value.get('gpu_uuids')
        if (not isinstance(uuids,list) or len(uuids)!=value['gpu_count']
                or any(not isinstance(x,str) or not x for x in uuids) or len(set(uuids))!=len(uuids)
                or not value.get('estimator_sha256')):
            raise ValueError('capture requires explicit unique GPU allocation and estimator identity')
    return dict(value, assumptions=assumptions)


def summarize(contract, binding, rows, *, age=None, completed=False, catching_up=False):
    result = {'status': 'unavailable', 'percent': None, 'range_percent': None,
              'unit': '%', 'source': 'useful_model_flops_over_logged_time_and_configured_peak'}
    if contract is None:
        return dict(result, reason='not_configured')
    if not rows:
        return dict(result, reason='not_started')
    if catching_up:
        return dict(result, reason='catching_up')
    if contract['config_sha256'] != binding.get('config_sha256'):
        return dict(result, reason='configuration_mismatch')
    if ('dataset_manifest_sha256' in contract and
            contract['dataset_manifest_sha256'] != binding.get('dataset_manifest_sha256')):
        return dict(result, reason='dataset_mismatch')
    environment = binding.get('environment', {})
    if isinstance(environment, dict):
        if ((environment.get('device') and environment['device'] != contract['gpu_model']) or
                (environment.get('precision') and str(environment['precision']).lower() != contract['precision'].lower())):
            return dict(result, reason='hardware_mismatch')
    recent = list(rows)[-100:]
    result.update(window_steps=len(recent), first_step=recent[0]['step'], last_step=recent[-1]['step'],
                  valid_timing_steps=sum(positive(r.get('seconds')) for r in recent),
                  profiled_steps=sum(r.get('profiled') is True for r in recent),
                  profiling_known_steps=sum(type(r.get('profiled')) is bool for r in recent))
    if result['valid_timing_steps'] != len(recent):
        return dict(result, reason='timing_incomplete')
    seconds = sum(r['seconds'] for r in recent)
    if not positive(seconds):
        return dict(result, reason='invalid_totals')
    mode = contract['mode']
    if mode == 'per_step':
        result['valid_flop_steps'] = sum(positive(r.get('model_flops')) for r in recent)
        if result['valid_flop_steps'] != len(recent):
            return dict(result, reason='flops_incomplete')
        total = sum(r['model_flops'] for r in recent)
    elif mode == 'constant':
        total = contract['flops_per_update'] * len(recent)
    else:
        total = None
    peak = contract['gpu_count'] * contract['peak_tflops_per_gpu'] * 1e12
    if total is not None and not positive(total):
        return dict(result, reason='invalid_totals')
    percent = (total / seconds / peak) * 100 if total is not None else None
    bounds = contract.get('flops_per_update_bounds')
    bounds = [v / seconds * len(recent) / peak * 100 for v in bounds] if bounds else None
    if ((percent is not None and (not positive(percent) or percent > 100)) or
            (bounds and (not all(positive(v) for v in bounds) or bounds[0] > 100))):
        return dict(result, reason='inconsistent_estimate')
    return dict(result, status='bounded' if mode == 'bounds' else 'estimated', mode=mode,
                percent=percent, range_percent=bounds, window_seconds=seconds,
                tflops=total / seconds / 1e12 if total is not None else None,
                temporal_status='completed' if completed else 'recent' if age is not None and age < 120 else 'stale',
                config_sha256=contract['config_sha256'],
                **{k: contract[k] for k in ('gpu_model', 'gpu_count', 'precision', 'sparsity',
                    'peak_tflops_per_gpu', 'peak_source', 'flop_source', 'timing_scope', 'assumptions')})
