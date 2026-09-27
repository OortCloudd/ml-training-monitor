import json
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


if __name__ == '__main__': unittest.main()
