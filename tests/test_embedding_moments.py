import copy
import json
import math
from pathlib import Path
import statistics
import tempfile
import time
import unittest
from unittest.mock import patch

from mlmonitor.config import load_config
from mlmonitor.embedding_moments import derive
from mlmonitor.reader import RunReader


MAPPINGS = {'ddof': 1, **{key: 'moments.' + key for key in (
    'per_dimension_std', 'n_samples', 'n_dimensions', 'norm_mean', 'norm_std')}}


def observed_moments(embeddings):
    norms = [math.sqrt(sum(value * value for value in row)) for row in embeddings]
    return {'n_samples': len(embeddings), 'n_dimensions': len(embeddings[0]),
            'per_dimension_std': [statistics.stdev(column) for column in zip(*embeddings)],
            'norm_mean': statistics.mean(norms), 'norm_std': statistics.stdev(norms)}


class MomentTests(unittest.TestCase):
    def test_nonzero_mean_agrees_with_direct_centered_energy(self):
        embeddings = [[2, 1], [2, 3], [4, 1], [4, 3]]
        result = derive(observed_moments(embeddings))
        self.assertAlmostEqual(result['total_variance'], 8 / 3)
        self.assertAlmostEqual(result['centered_feature_rms'], math.sqrt(4 / 3))
        # Mean embedding is [3, 2]; centered energy is 2, total energy is 15.
        self.assertAlmostEqual(result['centered_energy_fraction'], 2 / 15)

    def test_uniform_scale_preserves_energy_fraction(self):
        embeddings = [[2, 1], [2, 3], [4, 1], [4, 3]]
        before = derive(observed_moments(embeddings))
        after = derive(observed_moments([[7 * value for value in row] for row in embeddings]))
        self.assertAlmostEqual(after['total_variance'], 49 * before['total_variance'])
        self.assertAlmostEqual(after['centered_feature_rms'], 7 * before['centered_feature_rms'])
        self.assertAlmostEqual(after['centered_energy_fraction'], before['centered_energy_fraction'])

    def test_constant_nonzero_features_have_zero_centered_energy(self):
        result = derive(observed_moments([[3, 4]] * 4))
        self.assertEqual(result, {'total_variance': 0, 'centered_feature_rms': 0,
                                  'centered_energy_fraction': 0})

    def test_zero_total_energy_leaves_fraction_missing(self):
        result = derive(observed_moments([[0, 0]] * 4))
        self.assertEqual(result, {'total_variance': 0, 'centered_feature_rms': 0,
                                  'centered_energy_fraction': None})

    def test_missing_or_invalid_sources_remain_missing(self):
        valid = observed_moments([[2, 1], [2, 3], [4, 1], [4, 3]])
        for change in ({'per_dimension_std': None}, {'per_dimension_std': [0]},
                       {'per_dimension_std': [float('nan'), 1]}, {'per_dimension_std': [True, 1]},
                       {'per_dimension_std': [-1, 1]}, {'n_samples': 1}, {'n_samples': True},
                       {'n_dimensions': 0}, {'per_dimension_std': [1e308, 1e308]},
                       {'per_dimension_std': [10 ** 500, 1]}):
            with self.subTest(change=change):
                self.assertTrue(all(value is None for value in derive({**valid, **change}).values()))
        for change in ({'norm_mean': None}, {'norm_std': None}, {'norm_std': -1},
                       {'norm_std': float('inf')}, {'norm_mean': True}, {'norm_mean': 1e308},
                       {'norm_mean': 10 ** 500}):
            with self.subTest(change=change):
                result = derive({**valid, **change})
                self.assertIsNotNone(result['total_variance'])
                self.assertIsNotNone(result['centered_feature_rms'])
                self.assertIsNone(result['centered_energy_fraction'])


class MappingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'run'
        self.run.mkdir()

    def settings(self, contract=None, enabled=True):
        run = {'id': 'example', 'directory': 'run',
               'diagnostics': [{'key': 'total_variance', 'label': 'Total variance'}]}
        if enabled:
            run['embedding_moments'] = MAPPINGS if contract is None else contract
        config = self.root / 'monitor.json'
        config.write_text(json.dumps({'gpu_enabled': False, 'runs': [run]}))
        return load_config(config)['runs'][0]

    def test_mapping_contract_requires_explicit_sample_std_convention(self):
        for change in ({'ddof': 0}, {'ddof': True}, {'ddof': None},
                       {'n_samples': 'moments..count'}, {'norm_mean': ''}, {'norm_std': None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.settings({**MAPPINGS, **change})
        for removed in ('ddof', 'per_dimension_std'):
            contract = copy.deepcopy(MAPPINGS)
            contract.pop(removed)
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                self.settings(contract)

    def test_reader_derives_once_per_new_valid_observation_and_keeps_native_health(self):
        source = self.run / 'health_events.jsonl'
        event = {'update': 0, 'healthy': False, 'moments': observed_moments([[3, 4]] * 4)}
        source.write_text(json.dumps(event) + '\n')
        reader = RunReader(self.settings())
        with patch('mlmonitor.reader.derive_embedding_moments', wraps=derive) as calculation:
            first = reader.read(time.time())['health_points']
            self.assertEqual(calculation.call_count, 1)
            self.assertEqual(first[0]['total_variance'], 0)
            self.assertIs(first[0]['healthy'], False)
            self.assertNotIn('per_dimension_std', first[0])
            reader.read(time.time())
            self.assertEqual(calculation.call_count, 1)
            with source.open('a') as handle:
                handle.write(json.dumps(event) + '\n')  # duplicate update is ignored
                handle.write(json.dumps({**event, 'update': 1}) + '\n')
            self.assertEqual(len(reader.read(time.time())['health_points']), 2)
            self.assertEqual(calculation.call_count, 2)
            self.assertEqual(source.read_text().count('\n'), 3)

    def test_disabled_mapping_preserves_original_scalar_reader(self):
        event = {'update': 0, 'healthy': True, 'moments': observed_moments([[3, 4]] * 4)}
        (self.run / 'health_events.jsonl').write_text(json.dumps(event) + '\n')
        with patch('mlmonitor.reader.derive_embedding_moments') as calculation:
            result = RunReader(self.settings(enabled=False)).read(time.time())['health_points'][0]
            calculation.assert_not_called()
        self.assertEqual(result, {'step': 0, 'healthy': True, 'total_variance': None})


if __name__ == '__main__':
    unittest.main()
