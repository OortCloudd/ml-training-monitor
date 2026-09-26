import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mlmonitor.config import load_config
from mlmonitor.gpu_efficiency import window_summary, LIVE_METRICS
from mlmonitor.reader import RunReader
from mlmonitor.server import Sampler


class MetricQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'run').mkdir()
        path=self.root/'config.json'
        path.write_text(json.dumps({'gpu_enabled':False,'interval':60,'runs':[{'id':'test','directory':'run'}]}))
        self.settings=load_config(path)

    def write(self, rows):
        path=self.root/'run/training_telemetry.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return path

    def test_mean_window_counts_missing_values_and_profiled_updates(self):
        self.write([{'update':1,'timestamp':1000,'loss_components':{'loss':1},'update_seconds':.1,'profiled':False},
                    {'update':2,'timestamp':1001,'loss_components':{'loss':None},'update_seconds':.3,'profiled':True},
                    {'update':3,'timestamp':1002,'loss_components':{'loss':3}}])
        result=RunReader(self.settings['runs'][0]).read(1003)
        window=result['telemetry_window']
        self.assertEqual((window['first_update'],window['last_update'],window['records']),(1,3,3))
        self.assertEqual(window['valid_samples'],{'loss':2,'seconds':2,'gradient':0})
        self.assertEqual(window['profiled_records'],1)
        self.assertEqual(window['profiling_status_known_records'],2)
        self.assertEqual(result['loss'],2)
        self.assertAlmostEqual(result['seconds'],.2)
        self.assertEqual(result['freshness']['source'],'record_timestamp')
        self.assertEqual(result['freshness']['reference_at'],1002)
        self.assertEqual(result['freshness']['age_seconds'],1)

    def test_unknown_profiling_flag_is_not_counted_as_unprofiled(self):
        self.write([{'update':1,'profiled':'false'},{'update':2}])
        result=RunReader(self.settings['runs'][0]).read(time.time())
        self.assertEqual(result['telemetry_window']['profiling_status_known_records'],0)

    def test_file_time_fallback_and_future_source_clock_are_explicit(self):
        path=self.write([{'update':1}]);mtime=path.stat().st_mtime
        result=RunReader(self.settings['runs'][0]).read(mtime+20)
        self.assertEqual(result['freshness']['source'],'file_mtime')
        self.assertEqual(result['freshness']['reference_at'],mtime)
        self.write([{'update':1,'timestamp':2000}])
        result=RunReader(self.settings['runs'][0]).read(1000)
        self.assertEqual(result['freshness']['clock_ahead_seconds'],1000)

    def test_negative_durations_are_missing_but_zero_and_negative_loss_are_preserved(self):
        self.write([{'update':1,'update_seconds':-1,'loss_components':{'loss':-1}},
                    {'update':2,'update_seconds':0,'loss_components':{'loss':0}}])
        result=RunReader(self.settings['runs'][0]).read(time.time())
        self.assertEqual(result['telemetry_window']['valid_samples']['seconds'],1)
        self.assertEqual(result['seconds'],0)
        self.assertEqual(result['loss'],-.5)

    def test_hardware_window_has_counts_freshness_and_actual_interval(self):
        history=[{'time':t,'gpus':[{'index':4,'utilization':v,'power':None}]} for t,v in [(100,10),(160,None),(280,30)]]
        result=window_summary(history,4,300,seconds=300,interval=60)
        self.assertEqual(result['requested_interval_seconds'],60)
        self.assertEqual(result['median_interval_seconds'],90)
        self.assertEqual(result['last_sample_age_seconds'],20)
        self.assertEqual(result['stats']['utilization']['n'],2)
        self.assertEqual(result['stats']['power']['n'],0)
        self.assertIsNone(result['stats']['power']['mean'])
        self.assertEqual(LIVE_METRICS['memory_used']['unit'],'MiB')

    def test_publication_does_not_renew_the_hardware_observation_time(self):
        settings={**self.settings,'runs':[]}
        sampler=Sampler(settings)
        hardware={'cpu':10,'gpus':[{'index':0,'uuid':'GPU-test','utilization':20}]}
        with patch.object(sampler.hardware,'read',return_value=hardware),\
             patch('mlmonitor.server.time.time',side_effect=[1000,1002,1012]),\
             patch('mlmonitor.server.time.monotonic',side_effect=[10,22]):
            sampler.collect()
        result=sampler.snapshot
        self.assertEqual(result['timestamp'],1002)
        self.assertEqual(result['collection']['published_at'],1012)
        self.assertEqual(result['collection']['duration_seconds'],12)
        self.assertEqual(result['performance']['capabilities']['live_interval_seconds'],60)
        self.assertEqual(result['performance']['windows']['0']['last_sample_age_seconds'],10)
        self.assertIn('source',result['performance']['live_metric_definitions']['utilization'])


if __name__=='__main__':unittest.main()
