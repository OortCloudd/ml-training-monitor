"""Optional scheduler-record mapping; never controls a job or imports adapters."""
from .storage import field, read


def assign_runs(gpus, specs):
    processes = {}
    candidates = {}
    for spec in specs:
        runtime = spec.get('runtime')
        if runtime is None:
            continue
        raw = read(runtime['path'])
        values = {k: field(raw, v) for k, v in runtime['fields'].items()}
        if (values['run_id'] != runtime['run_id_value'] or values['status'] != runtime['running_value']
                or type(values['pid']) is not int or values['pid'] < 1
                or not isinstance(values['gpu_uuid'], str)):
            continue
        candidates.setdefault(values['gpu_uuid'], []).append((spec, values['pid']))
    for gpu in gpus:
        gpu.update(run_id='', run_name='Charge non identifiée', run_assignment='unverified')
        matches = candidates.get(gpu['uuid'], [])
        if len(matches) != 1:
            continue
        spec, pid = matches[0]
        if gpu.get('compute_pids') != [pid]:
            continue
        gpu.update(run_id=spec['id'], run_name=spec['name'], run_assignment='scheduler_and_sole_compute_pid')
        processes[spec['id']] = pid
    return processes
