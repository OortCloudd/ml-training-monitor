import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlmonitor import Monitor
from mlmonitor.config import load_config
from mlmonitor.reader import RunReader


class RunStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name):
        root = self.root / name
        root.mkdir()
        config = root / 'monitor.json'
        config.write_text(json.dumps({'gpu_enabled': False, 'runs': [
            {'id': 'fixture', 'directory': 'run'}]}))
        monitor = Monitor(root / 'run', run_id='fixture', config={'fixture': True}, target_updates=100)
        return monitor, RunReader(load_config(config)['runs'][0])

    @staticmethod
    def step(monitor, update, timestamp):
        with patch('mlmonitor.recorder.time.time', return_value=timestamp):
            with monitor.step(update) as observation:
                observation.record(loss=1, step_seconds=1)

    def test_recorder_stop_states_override_recent_telemetry_and_eta(self):
        for status, label in [('PAUSED', 'En pause'), ('FAILED', 'Échec'), ('COMPLETE', 'Terminé')]:
            with self.subTest(status=status):
                monitor, reader = self.fixture(status)
                self.step(monitor, 1, 1000)
                reader.read(1000)
                self.step(monitor, 2, 1060)
                self.assertEqual(reader.read(1060)['eta'], 5880)
                monitor.close(status)
                result = reader.read(1060)
                self.assertEqual(result['producer_status'], status)
                self.assertEqual(result['status'], label)
                self.assertIsNone(result['eta'])
                self.assertEqual(result['step'], 2)
                self.assertEqual(result['freshness']['age_seconds'], 0)
                self.assertEqual(reader.read(2000)['status'], label)

    def test_resume_requires_a_new_progress_window(self):
        for status in ('PAUSED', 'FAILED'):
            with self.subTest(status=status):
                monitor, reader = self.fixture(status)
                self.step(monitor, 1, 1000)
                reader.read(1000)
                self.step(monitor, 2, 1060)
                self.assertIsNotNone(reader.read(1060)['eta'])
                monitor.close(status)
                reader.read(1060)
                resumed = Monitor(monitor.directory, run_id='fixture', config={'fixture': True},
                                  target_updates=100, start_step=2)
                result = reader.read(1070)
                self.assertEqual(result['producer_status'], 'RUNNING')
                self.assertEqual(result['status'], 'Télémétrie récente')
                self.assertIsNone(result['eta'])
                self.step(resumed, 3, 1100)
                self.assertIsNone(reader.read(1100)['eta'])
                self.step(resumed, 4, 1130)
                self.assertEqual(reader.read(1130)['eta'], 2880)

    def test_failure_before_first_step_is_not_reported_as_unstarted(self):
        monitor, reader = self.fixture('early_failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            with monitor.step(1):
                raise RuntimeError('synthetic failure')
        result = reader.read(1000)
        self.assertEqual(result['producer_status'], 'FAILED')
        self.assertEqual(result['status'], 'Échec')
        self.assertEqual(result['window'], 0)
        self.assertIsNone(result['eta'])

    def test_unknown_or_missing_state_keeps_telemetry_semantics(self):
        monitor, reader = self.fixture('existing_logs')
        state = monitor.directory / 'state.json'
        state.unlink()
        result = reader.read(1000)
        self.assertIsNone(result['producer_status'])
        self.assertEqual(result['status'], 'Pas démarré')
        self.step(monitor, 1, 1000)
        state.write_text(json.dumps({'status': 'OTHER'}))
        self.assertEqual(reader.read(1000)['status'], 'Télémétrie récente')
        result = reader.read(1200)
        self.assertIsNone(result['producer_status'])
        self.assertEqual(result['status'], 'Télémétrie ancienne')


if __name__ == '__main__':
    unittest.main()
