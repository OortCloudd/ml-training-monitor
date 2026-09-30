import json
import copy
from pathlib import Path
import tempfile
import unittest

from mlmonitor.config import load_config
from mlmonitor.runtime import assign_runs


class RuntimeTests(unittest.TestCase):
    def test_optional_mapping_requires_scheduler_and_exclusive_compute_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'runs': [{'id': 'test', 'directory': '.', 'runtime': {
                'path': 'queue.json', 'fields': {'run_id': 'active_method', 'pid': 'active_pid'},
                'run_id_value': 'TEST'}}]}
            (root/'config.json').write_text(json.dumps(config))
            specs = load_config(root/'config.json')['runs']
            def check(record, pids):
                (root/'queue.json').write_text(json.dumps(record))
                gpus = [{'uuid': 'GPU-test', 'compute_pids': pids}]
                result = assign_runs(gpus, specs)
                return result, gpus[0]
            valid = {'active_method': 'TEST', 'active_pid': 123, 'gpu_uuid': 'GPU-test', 'status': 'RUNNING'}
            result, gpu = check(valid, [123])
            self.assertEqual(result, {'test': 123})
            self.assertEqual(gpu['run_id'], 'test')
            for pids in (None, [], [456], [123, 456]):
                result, gpu = check(valid, pids)
                self.assertFalse(result)
                self.assertEqual(gpu['run_assignment'], 'unverified')
            for changes in ({'active_method': 'OTHER'}, {'status': 'COMPLETE'}, {'active_pid': True}):
                self.assertFalse(check(dict(valid, **changes), [123])[0])
            gpus = [{'uuid': 'GPU-test', 'compute_pids': [123]}]
            (root/'queue.json').write_text(json.dumps(valid))
            self.assertFalse(assign_runs(gpus, specs + specs))

    def distributed_specs(self, root, fields=None):
        runtime = {'path': 'queue.json', 'run_id_value': 'TEST', 'fields': fields or {
            'run_id': 'active_method', 'workers': 'workers', 'coordinator_pid': 'coordinator_pid'}}
        config = {'runs': [{'id': 'test', 'name': 'One logical run', 'directory': '.', 'runtime': runtime}]}
        (root/'config.json').write_text(json.dumps(config))
        return load_config(root/'config.json')['runs']

    def distributed_record(self):
        return {'active_method': 'TEST', 'status': 'RUNNING', 'coordinator_pid': 123,
                'workers': [{'pid': 123, 'gpu_uuid': 'GPU-a', 'rank': 0},
                            {'pid': 456, 'gpu_uuid': 'GPU-b', 'rank': 1}]}

    def check_distributed(self, root, specs, record, gpus=None):
        (root/'queue.json').write_text(json.dumps(record))
        if gpus is None:
            gpus = [{'uuid': 'GPU-a', 'compute_pids': [123]}, {'uuid': 'GPU-b', 'compute_pids': [456]}]
        result = assign_runs(gpus, specs)
        return result, gpus

    def assert_unverified(self, result, gpus):
        self.assertFalse(result)
        self.assertTrue(all(g['run_id'] == '' and g['run_assignment'] == 'unverified' for g in gpus))

    def test_worker_allocation_assigns_both_devices_to_one_run_and_coordinator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = self.distributed_specs(root)
            record = self.distributed_record()
            for coordinator in (123, 456):
                record['coordinator_pid'] = coordinator
                record['workers'].reverse()
                result, gpus = self.check_distributed(root, specs, record)
                self.assertEqual(result, {'test': coordinator})
                self.assertEqual([g['run_id'] for g in gpus], ['test', 'test'])
                self.assertTrue(all(g['run_assignment'] == 'scheduler_and_complete_worker_allocation' for g in gpus))

    def test_worker_allocation_is_atomic_when_device_ownership_is_missing_or_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = self.distributed_specs(root)
            record = self.distributed_record()
            for pids in (None, [], [999], [456, 999]):
                gpus = [{'uuid': 'GPU-a', 'compute_pids': [123]}, {'uuid': 'GPU-b', 'compute_pids': pids}]
                with self.subTest(pids=pids):
                    self.assert_unverified(*self.check_distributed(root, specs, record, gpus))
            self.assert_unverified(*self.check_distributed(root, specs, record,
                [{'uuid': 'GPU-a', 'compute_pids': [123]}]))
            self.assert_unverified(*self.check_distributed(root, specs + specs, record))
            # An overlapping scheduler claim invalidates the whole multi-worker
            # allocation, including its otherwise exclusively owned GPU-a.
            (root/'other.json').write_text(json.dumps({'run_id': 'other', 'status': 'RUNNING',
                                                     'pid': 456, 'gpu_uuid': 'GPU-b'}))
            other_config = {'runs': [{'id': 'other', 'directory': '.', 'runtime': {'path': 'other.json'}}]}
            (root/'other-config.json').write_text(json.dumps(other_config))
            other = load_config(root/'other-config.json')['runs']
            self.assert_unverified(*self.check_distributed(root, specs + other, record))

    def test_worker_record_rejects_duplicate_or_incomplete_identities_and_bad_coordinator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = self.distributed_specs(root)
            base = self.distributed_record()
            invalid = []
            for value in (None, [], {}, 'workers'):
                invalid.append(dict(base, workers=value))
            invalid.append(dict(base, workers=[base['workers'][0]]))
            for value in (None, True, 0, 999, '123'):
                invalid.append(dict(base, coordinator_pid=value))
            for key, value in (('pid', 123), ('pid', True), ('pid', 0), ('gpu_uuid', 'GPU-a'),
                               ('gpu_uuid', ''), ('rank', 0), ('rank', 2), ('rank', True)):
                record = copy.deepcopy(base)
                record['workers'][1][key] = value
                invalid.append(record)
            record = copy.deepcopy(base)
            del record['workers'][1]['rank']
            invalid.append(record)
            record = copy.deepcopy(base)
            record['workers'].append({'pid': 789, 'gpu_uuid': 'GPU-c', 'rank': 2})
            invalid.append(record)
            invalid.extend([dict(base, active_method='OTHER'), dict(base, status='PAUSED')])
            # Configured worker mode must never fall back to the valid legacy
            # identity when its explicit allocation is unavailable.
            invalid.append({'active_method': 'TEST', 'status': 'RUNNING', 'pid': 123, 'gpu_uuid': 'GPU-a'})
            for record in invalid:
                with self.subTest(record=record):
                    self.assert_unverified(*self.check_distributed(root, specs, record))

    def test_workers_are_opt_in_and_support_nested_scheduler_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self.distributed_record()
            fields = {'run_id': 'job.method', 'status': 'job.state',
                      'workers': 'job.allocation', 'coordinator_pid': 'job.writer'}
            specs = self.distributed_specs(root, fields)
            nested = {'job': {'method': 'TEST', 'state': 'RUNNING', 'allocation': record['workers'], 'writer': 123}}
            self.assertEqual(self.check_distributed(root, specs, nested)[0], {'test': 123})
            legacy_config = {'runs': [{'id': 'test', 'directory': '.', 'runtime': {'path': 'queue.json'}}]}
            (root/'legacy.json').write_text(json.dumps(legacy_config))
            legacy = load_config(root/'legacy.json')['runs']
            record.update(run_id='test', pid=123, gpu_uuid='GPU-a', workers='ignored by legacy mapping')
            result, gpus = self.check_distributed(root, legacy, record)
            self.assertEqual(result, {'test': 123})
            self.assertEqual(gpus[0]['run_assignment'], 'scheduler_and_sole_compute_pid')
            self.assertEqual(gpus[1]['run_assignment'], 'unverified')

    def test_worker_and_coordinator_mappings_must_be_supplied_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fields in ({'workers': 'workers'}, {'coordinator_pid': 'writer'},
                           {'workers': 'workers', 'coordinator_pid': None}):
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    self.distributed_specs(root, fields)


if __name__ == '__main__': unittest.main()
