"""Optional scheduler-record mapping; never controls a job or imports adapters."""
from .storage import field, read


def worker_allocation(values):
    """Validate an explicit local allocation before matching physical processes."""
    workers = values.get('workers')
    coordinator = values.get('coordinator_pid')
    if not isinstance(workers, list) or len(workers) < 2:
        return None
    for worker in workers:
        if (not isinstance(worker, dict) or set(worker) != {'pid', 'gpu_uuid', 'rank'}
                or type(worker['pid']) is not int or worker['pid'] < 1
                or not isinstance(worker['gpu_uuid'], str) or not worker['gpu_uuid']
                or type(worker['rank']) is not int or worker['rank'] < 0):
            return None
    count = len(workers)
    if (len({w['pid'] for w in workers}) != count
            or len({w['gpu_uuid'] for w in workers}) != count
            or {w['rank'] for w in workers} != set(range(count))
            or type(coordinator) is not int
            or coordinator not in {w['pid'] for w in workers}):
        return None
    return workers, coordinator


def assign_runs(gpus, specs):
    processes = {}
    candidates = {}
    allocations = []
    for spec in specs:
        runtime = spec.get('runtime')
        if runtime is None:
            continue
        raw = read(runtime['path'])
        values = {k: field(raw, v) for k, v in runtime['fields'].items()}
        if values['run_id'] != runtime['run_id_value'] or values['status'] != runtime['running_value']:
            continue
        multi = 'workers' in runtime['fields']
        if multi:
            allocation = worker_allocation(values)
            if allocation is None:
                continue
            workers, coordinator = allocation
        else:
            if (type(values['pid']) is not int or values['pid'] < 1
                    or not isinstance(values['gpu_uuid'], str)):
                continue
            workers = [{'pid': values['pid'], 'gpu_uuid': values['gpu_uuid']}]
            coordinator = values['pid']
        allocation_index = len(allocations)
        allocations.append((spec, workers, coordinator, multi))
        for worker in workers:
            candidates.setdefault(worker['gpu_uuid'], []).append(allocation_index)
    for gpu in gpus:
        gpu.update(run_id='', run_name='Charge non identifiée', run_assignment='unverified')
    for spec, workers, coordinator, multi in allocations:
        matched = []
        for worker in workers:
            devices = [gpu for gpu in gpus if gpu['uuid'] == worker['gpu_uuid']]
            if (len(candidates[worker['gpu_uuid']]) != 1 or len(devices) != 1
                    or devices[0].get('compute_pids') != [worker['pid']]):
                break
            matched.append(devices[0])
        if len(matched) != len(workers):
            continue
        for gpu in matched:
            gpu.update(run_id=spec['id'], run_name=spec['name'],
                       run_assignment='scheduler_and_complete_worker_allocation' if multi
                       else 'scheduler_and_sole_compute_pid')
        processes[spec['id']] = coordinator
    return processes
