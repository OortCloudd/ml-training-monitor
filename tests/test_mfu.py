import json
from pathlib import Path
import tempfile
import unittest

from mlmonitor import Monitor
from mlmonitor.config import load_config
from mlmonitor.mfu import summarize, validate_contract
from mlmonitor.reader import RunReader
from mlmonitor.storage import digest


def contract(**changes):
    return dict(config_sha256='a'*64, gpu_model='Test accelerator', gpu_count=2,
                precision='bf16', sparsity='dense', peak_tflops_per_gpu=100,
                peak_source='Synthetic test ceiling, not a hardware measurement',
                flop_source='Synthetic test counts', timing_scope='Whole optimizer update',
                mode='per_step', **changes)


class MFUTests(unittest.TestCase):
    def test_variable_work_uses_ratio_of_totals_and_allocated_devices(self):
        c = validate_contract(contract())
        rows = [{'step': 1, 'seconds': 1, 'model_flops': 100e12, 'profiled': False},
                {'step': 2, 'seconds': 3, 'model_flops': 100e12, 'profiled': True}]
        r = summarize(c, {'config_sha256': 'a'*64}, rows, age=0)
        self.assertEqual(r['percent'], 25)
        self.assertEqual(r['tflops'], 50)
        self.assertEqual(r['profiled_steps'], 1)
        self.assertEqual(r['profiling_known_steps'], 2)
        self.assertEqual(r['temporal_status'], 'recent')

    def test_constant_bounds_missing_and_stale_have_distinct_meanings(self):
        c = contract(); c.update(mode='constant', flops_per_update=100e12)
        c = validate_contract(c)
        rows = [{'step': 1, 'seconds': 2}]
        self.assertEqual(summarize(c, {'config_sha256': 'a'*64}, rows, age=200)['temporal_status'], 'stale')
        c.update(mode='bounds', flops_per_update=None, flops_per_update_bounds=[50e12, 200e12])
        r = summarize(validate_contract(c), {'config_sha256': 'a'*64}, rows, completed=True)
        self.assertIsNone(r['percent'])
        self.assertEqual(r['range_percent'], [12.5, 50])
        self.assertEqual(r['temporal_status'], 'completed')
        self.assertEqual(summarize(None, {}, rows)['reason'], 'not_configured')

    def test_missing_counts_and_bad_times_cannot_become_zero(self):
        c = validate_contract(contract())
        for bad in (None, 0, -1, float('nan'), float('inf'), True):
            for key in ('seconds', 'model_flops'):
                row = {'step': 1, 'seconds': 2, 'model_flops': 100e12}; row[key] = bad
                r = summarize(c, {'config_sha256': 'a'*64}, [row])
                self.assertEqual(r['status'], 'unavailable')
                self.assertIsNone(r['percent'])
        r = summarize(c, {'config_sha256': 'a'*64}, [{'step': 1, 'seconds': .01, 'model_flops': 100e12}])
        self.assertEqual(r['reason'], 'inconsistent_estimate')

    def test_binding_change_and_backlog_invalidate(self):
        c = validate_contract(contract(dataset_manifest_sha256='b'*64))
        rows = [{'step': 1, 'seconds': 1, 'model_flops': 1}]
        binding = {'config_sha256': 'a'*64, 'dataset_manifest_sha256': 'b'*64}
        self.assertEqual(summarize(c, dict(binding, config_sha256='c'*64), rows)['reason'], 'configuration_mismatch')
        self.assertEqual(summarize(c, dict(binding, dataset_manifest_sha256='c'*64), rows)['reason'], 'dataset_mismatch')
        self.assertEqual(summarize(c, dict(binding, environment={'precision': 'fp32'}), rows)['reason'], 'hardware_mismatch')
        self.assertEqual(summarize(c, binding, rows, catching_up=True)['reason'], 'catching_up')

    def test_last_100_counts_not_entire_history(self):
        c = validate_contract(contract())
        rows = [{'step': i, 'seconds': 1, 'model_flops': 20e12} for i in range(1, 151)]
        rows[0]['seconds'] = None
        r = summarize(c, {'config_sha256': 'a'*64}, rows)
        self.assertEqual((r['first_step'], r['last_step'], r['window_steps']), (51, 150, 100))
        self.assertAlmostEqual(r['percent'], 10)
        self.assertEqual(r['profiling_known_steps'], 0)

    def test_invalid_contracts_are_rejected_before_serving(self):
        for changes in ({'gpu_count': True}, {'peak_tflops_per_gpu': -1}, {'config_sha256': 'bad'},
                        {'mode': 'mystery'}, {'sparsity': 'unknown'}, {'timing_scope': ''},
                        {'mode': 'bounds', 'flops_per_update_bounds': [4, 2]},
                        {'mode': 'constant', 'flops_per_update': 2, 'flops_per_update_bounds': [3, 4]}):
            c = contract(); c.update(changes)
            with self.assertRaises(ValueError): validate_contract(c)

    def test_optional_scalar_hook_through_real_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = {'model': 'test'}
            monitor = Monitor(root/'run', run_id='run', config=config)
            with monitor.step(1) as row:
                row.record(loss=1, step_seconds=2, model_flops=100e12)
            monitor.close()
            c = contract(); c['config_sha256'] = digest(config)
            path = root/'monitor.json'
            path.write_text(json.dumps({'gpu_enabled': False, 'runs': [
                {'id': 'run', 'directory': 'run', 'mfu': c}]}))
            run = RunReader(load_config(path)['runs'][0]).read(0)
            self.assertEqual(run['mfu']['percent'], 25)
            self.assertEqual(run['loss'], 1)
            # A caller's tensor-like object must never trigger device conversion.
            class TensorLike:
                def item(self): raise AssertionError('Unexpected tensor conversion')
            monitor = Monitor(root/'other', run_id='other', config=config)
            with monitor.step(1) as row: row.record(model_flops=TensorLike())
            monitor.close()
            payload = json.loads((root/'other/training_telemetry.jsonl').read_text())
            self.assertNotIn('model_flops', payload)

    def test_existing_log_mapping_uses_same_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'run_bindings.json').write_text(json.dumps({'config_sha256': 'a'*64}))
            (root/'training_telemetry.jsonl').write_text(json.dumps(
                {'update': 1, 'update_seconds': 2, 'compute': {'useful': 100e12}})+'\n')
            (root/'config.json').write_text(json.dumps({'runs': [{'id': 'run', 'directory': '.',
                'mfu': contract(), 'fields': {'model_flops': 'compute.useful'}}]}))
            run = RunReader(load_config(root/'config.json')['runs'][0]).read(0)
            self.assertEqual(run['mfu']['percent'], 25)


if __name__ == '__main__': unittest.main()
