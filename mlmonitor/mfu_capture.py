"""Dated periodic MFU: counted work, matched unprofiled timing and explicit limits."""
import json
from .storage import finite
from .uncertainty import rate_interval


def summarize_capture(path,contract,binding,now):
    empty={'status':'unavailable','percent':None,'range_percent':None,'reason':'capture_missing'}
    if contract is None:return empty
    if contract['config_sha256']!=binding.get('config_sha256'):
        return dict(empty,reason='configuration_mismatch')
    if contract.get('dataset_manifest_sha256') and contract['dataset_manifest_sha256']!=binding.get('dataset_manifest_sha256'):
        return dict(empty,reason='dataset_mismatch')
    env=binding.get('environment',{})
    if ((env.get('device') and env['device']!=contract['gpu_model']) or
            (env.get('precision') and str(env['precision']).lower()!=contract['precision'].lower())):
        return dict(empty,reason='hardware_mismatch')
    try:
        if path.stat().st_size>2_000_000:raise ValueError('oversized capture')
        c=json.loads(path.read_bytes())
        if (c['schema']!='mfu-periodic-capture-v1' or c['valid'] is not True
                or c['config_sha256']!=contract['config_sha256']
                or c['estimator_sha256']!=contract['estimator_sha256']
                or c['gpu_uuids']!=contract['gpu_uuids']
                or c['gpu_model']!=contract['gpu_model']
                or c['precision']!=contract['precision'].lower()
                or c['sparsity']!=contract['sparsity']
                or c['profile_timing_used_for_mfu'] is not False
                or c['dispatcher_flops']!=c['work']['executed_model_flops']):
            return dict(empty,reason='capture_identity_or_count_mismatch')
        ops=c.get('operator_flops',{});dtypes=c.get('flops_by_dtype',{})
        expected_dtype={'bf16':'torch.bfloat16','fp16':'torch.float16','fp32':'torch.float32','fp64':'torch.float64'}.get(contract['precision'].lower())
        if (not ops or any(finite(v) is None or v<0 for v in ops.values())
                or sum(ops.values())!=c['dispatcher_flops']
                or expected_dtype is None or set(dtypes)!={expected_dtype} or sum(dtypes.values())!=c['dispatcher_flops']
                or c['work'].get('update')!=c['milestone']
                or c['work'].get('config_sha256')!=c['config_sha256']
                or c['nsight_join']!={'run_id':c['run_id'],'config_sha256':c['config_sha256'],'milestone':c['milestone']}):
            return dict(empty,reason='capture_identity_or_count_mismatch')
        stamp=finite(c['captured_at'])
        if stamp is None or stamp>now:return dict(empty,reason='invalid_capture_time')
        rows=c['baseline_steps']
        if not rows:return dict(empty,reason='capture_timing_missing')
        for r in rows:
            if (r.get('valid') is not True or r.get('profiled') is not False
                    or r.get('config_sha256')!=c['config_sha256']
                    or r.get('estimator_sha256')!=c['estimator_sha256']
                    or r.get('gpu_uuids')!=c['gpu_uuids']
                    or not isinstance(r.get('telemetry_record_sha256'),str)
                    or len(r['telemetry_record_sha256'])!=64
                    or any(finite(r.get(k)) is None or r[k]<=0 for k in ('model_flops','executed_model_flops','update_seconds'))
                    or r['executed_model_flops']<r['model_flops']):
                return dict(empty,reason='capture_timing_unverified')
        if any(b['update']!=a['update']+1 for a,b in zip(rows,rows[1:])) or rows[-1]['update']!=c['milestone']-1:
            return dict(empty,reason='capture_window_gap')
        work=sum(r['model_flops'] for r in rows);seconds=sum(r['update_seconds'] for r in rows)
        peak=contract['gpu_count']*contract['peak_tflops_per_gpu']*1e12
        percent=100*work/seconds/peak
        if percent>100:return dict(empty,reason='inconsistent_estimate')
        rates=[100*r['model_flops']/r['update_seconds']/peak for r in rows]
        ci=dict(rate_interval(tuple((r['model_flops'],r['update_seconds']) for r in rows)))
        if ci['status']=='available':
            ci['lower_percent']=100*ci.pop('lower_rate')/peak;ci['upper_percent']=100*ci.pop('upper_rate')/peak
        return dict(contract,status='estimated',mode='capture',percent=percent,range_percent=None,
                    source='periodic_observed_operator_capture',temporal_status='capture',
                    captured_at=stamp,capture_age_seconds=now-stamp,capture_update=c['milestone'],
                    capture_reason=c.get('capture_reason','milestone'),count_basis='observed_shapes_dispatcher_validated',
                    first_step=rows[0]['update'],last_step=rows[-1]['update'],window_steps=len(rows),
                    window_seconds=seconds,tflops=work/seconds/1e12,
                    observed_range_percent=[min(rates),max(rates)],observed_range_kind='descriptive_not_confidence',
                    confidence_interval=ci,profiled_steps=0,profiling_known_steps=len(rows),
                    counted_profile_flops=c['work']['model_flops'],counted_executed_flops=c['dispatcher_flops'],
                    recompute_flops=c['work']['recompute_flops'],
                    arithmetic_scope=c['arithmetic_scope'],systematic_limit=c['systematic_limit'],
                    nsight_join=c['nsight_join'],current_live_estimate=False)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,ZeroDivisionError):
        return dict(empty,reason='capture_invalid')


def reduce_data_parallel_step(records, *, run_id, config_sha256, step, gpu_uuids, step_seconds):
    """Sum unique rank-local work once; use one coordinator duration, not sum(rank time)."""
    if not isinstance(gpu_uuids,list) or not gpu_uuids or len(set(gpu_uuids))!=len(gpu_uuids):
        raise ValueError('unique device allocation required')
    if finite(step_seconds) is None or step_seconds<=0:raise ValueError('coordinator duration required')
    if len(records)!=len(gpu_uuids):raise ValueError('missing rank')
    seen=set();devices=set();total=0.;times=[]
    for r in records:
        rank=r.get('rank');uuid=r.get('gpu_uuid');count=finite(r.get('model_flops'));seconds=finite(r.get('seconds'))
        if (type(rank) is not int or rank not in range(len(gpu_uuids)) or rank in seen
                or uuid not in gpu_uuids or uuid in devices):raise ValueError('duplicate or foreign rank/device')
        if (r.get('run_id')!=run_id or r.get('config_sha256')!=config_sha256 or r.get('step')!=step
                or r.get('world_size')!=len(gpu_uuids) or r.get('work_scope')!='rank_local_unique_samples'):
            raise ValueError('unaligned or already-global work')
        if count is None or count<=0 or seconds is None or seconds<=0:raise ValueError('incomplete work/time')
        if seconds>step_seconds*(1+1e-6):raise ValueError('rank exceeds coordinator window')
        seen.add(rank);devices.add(uuid);total+=count;times.append(seconds)
    return {'step':step,'seconds':step_seconds,'model_flops':total,'gpu_uuids':list(gpu_uuids),
            'rank_count':len(seen),'rank_time_spread_seconds':max(times)-min(times)}
