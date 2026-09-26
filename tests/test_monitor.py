from contextlib import contextmanager
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from mlmonitor import Monitor
from mlmonitor.config import load_config
from mlmonitor.reader import JsonLines, RunReader
from mlmonitor.recorder import due, next_capture
from mlmonitor.server import Sampler
from mlmonitor.storage import read
from mlmonitor.worker import CapturePlan, process_request, execute


class Base(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def monitor(self, **kwargs):
        return Monitor(self.root/'run',run_id='test_run',config={'model':'unit fixture'},**kwargs)

    def settings(self, **run_values):
        path=self.root/'monitor.json'
        path.write_text(json.dumps({'gpu_enabled':False,'runs':[{'id':'test_run','directory':'run',**run_values}]}))
        return load_config(path)


class CollectionTests(Base):
    def test_light_monitor_to_dashboard_includes_phases_operations_and_diagnostics(self):
        monitor=self.monitor(target_updates=10,details={'Model':'Unit fixture'})
        monitor.diagnostic(0,{'accuracy':.25},healthy=True)
        with monitor.step(1) as observation:
            with monitor.phase('input_wait'):pass
            with monitor.phase('forward_host'):pass
            observation.record(loss=.5,gradient_norm=.2,step_seconds=.01)
        with monitor.operation('evaluation'):pass
        monitor.flush()
        sampler=Sampler(self.settings(diagnostics=[{'key':'accuracy','label':'Validation accuracy'}]))
        with patch('mlmonitor.hardware.subprocess.run') as process:sampler.collect()
        process.assert_not_called()
        run=sampler.snapshot['runs'][0]
        self.assertEqual(run['step'],1);self.assertEqual(run['loss'],.5)
        self.assertEqual(run['health_points'][0]['step'],0)
        self.assertEqual(run['health_points'][0]['accuracy'],.25)
        self.assertIn('forward_host',run['light_profile']['phases'])
        self.assertIn('evaluation',run['operations']['operations'])
        self.assertEqual(run['bottleneck']['status'],'disabled')
        self.assertEqual(run['network']['fields'],[('Model','Unit fixture')])

    def test_no_profiler_factory_or_rng_change_in_light_mode(self):
        random.seed(14);before=random.getstate()
        def forbidden():raise AssertionError('profiler factory called')
        monitor=self.monitor(capture_factory=forbidden)
        with monitor.step(1):
            with monitor.phase('backward_host'):pass
        self.assertEqual(before,random.getstate())
        self.assertFalse((self.root/'run/profile.json').exists())

    def test_training_exception_survives_observer_cleanup(self):
        monitor=self.monitor()
        with self.assertRaisesRegex(RuntimeError,'training error'):
            with monitor.step(1):raise RuntimeError('training error')
        self.assertEqual(read(self.root/'run/state.json')['status'],'FAILED')
        self.assertFalse((self.root/'run/training_telemetry.jsonl').exists())

    def test_monitor_io_failure_does_not_replace_training_work(self):
        monitor=self.monitor()
        with patch('mlmonitor.recorder.atomic',side_effect=OSError('synthetic storage error')):
            with monitor.step(1) as observation:observation.record(loss=1)
        self.assertEqual(monitor.last_update,1)
        self.assertEqual(monitor.error,'OSError')

    def test_duplicate_steps_and_config_reuse_are_rejected(self):
        monitor=self.monitor()
        with monitor.step(1):pass
        with self.assertRaises(ValueError):
            with monitor.step(1):pass
        with self.assertRaises(ValueError):
            Monitor(self.root/'run',run_id='test_run',config={'different':True})

    def test_summary_writes_are_rate_limited_and_flush_retains_last_step(self):
        monitor=self.monitor(publish_seconds=3600)
        with monitor.step(1):pass
        before=(self.root/'run/phase_timings.json').read_bytes()
        with monitor.step(2):pass
        self.assertEqual(before,(self.root/'run/phase_timings.json').read_bytes())
        monitor.flush()
        self.assertEqual(read(self.root/'run/phase_timings.json')['last_update'],2)

    def test_all_history_and_partial_lines_are_preserved(self):
        folder=self.root/'run';folder.mkdir()
        log=folder/'training_telemetry.jsonl'
        log.write_text(''.join(json.dumps({'update':i,'loss_components':{'loss':i}})+'\n' for i in range(1,2001))+'{"update":2001}')
        reader=RunReader(self.settings()['runs'][0]);out=reader.read(time.time())
        self.assertEqual(out['points'][0]['step'],1)
        self.assertEqual(out['points'][1]['loss'],50.5)
        self.assertEqual(out['points'][-1]['step'],2000)
        self.assertEqual(out['window'],1200)
        with log.open('a') as handle:handle.write('\n')
        self.assertEqual(reader.read(time.time())['step'],2001)
        log.write_text('{"update":1}\n')
        self.assertEqual(reader.read(time.time())['step'],1)

    def test_existing_log_mappings_and_missing_values(self):
        folder=self.root/'run';folder.mkdir()
        (folder/'custom.jsonl').write_text('{"global_step":1,"metrics":{"objective":0.2},"gradient":true,"private_field":"never export"}\n')
        reader=RunReader(self.settings(files={'telemetry':'custom.jsonl'},fields={'step':'global_step','loss':'metrics.objective','gradient':'gradient'})['runs'][0])
        out=reader.read(time.time())
        self.assertEqual(out['loss'],.2);self.assertIsNone(out['gradient']);self.assertIsNone(out['seconds'])
        self.assertNotIn('never export',json.dumps(out))

    def test_diagnostics_keep_zero_and_no_unconfigured_arrays(self):
        monitor=self.monitor()
        monitor.diagnostic(0,{'rank':None,'array':[1,2],'accuracy':0},healthy=False)
        reader=RunReader(self.settings(diagnostics=[{'key':'accuracy','label':'Accuracy'}])['runs'][0])
        point=reader.read(time.time())['health_points'][0]
        self.assertEqual(point,{'step':0,'healthy':False,'accuracy':0})

    def test_existing_phase_timers_need_no_new_training_hook(self):
        folder=self.root/'run';folder.mkdir()
        (folder/'training_telemetry.jsonl').write_text('{"update":1,"prepared_input_wait_seconds":0.1}\n{"update":2}\n')
        sampler=Sampler(self.settings());sampler.collect()
        phase=sampler.snapshot['runs'][0]['light_profile']['phases']['input_wait']
        self.assertEqual(phase,{'seconds':.1,'calls':1,'steps':1})

    def test_same_inode_rewrite_and_malformed_lines(self):
        path=self.root/'records';path.write_text('{"update":100}\n')
        stream=JsonLines();self.assertEqual(stream.poll(path)[0]['update'],100)
        generation=stream.generation
        path.write_text('{"update":1,"extra":"longer"}\nnot json\n[]\n')
        self.assertEqual(stream.poll(path)[0]['update'],1)
        self.assertGreater(stream.generation,generation);self.assertEqual(stream.skipped,2)


class ProfilingTests(Base):
    def test_resume_and_exact_schedule(self):
        self.assertEqual([i for i in range(1,11) if due(i,every=3)],[3,6,9])
        self.assertFalse(due(0,every=3));self.assertEqual(next_capture(6,every=3),9)
        calls=[]
        def unavailable():calls.append(1);raise RuntimeError('no profiler')
        monitor=self.monitor(profile_every=3,start_step=3,capture_factory=unavailable)
        with monitor.step(4):pass
        self.assertEqual(calls,[])
        with monitor.step(6):pass
        self.assertEqual(calls,[1])
        with monitor.step(9):pass
        self.assertEqual(calls,[1]);self.assertTrue(monitor.profile_disabled)

    def test_capture_publishes_partition_history_and_next_milestone(self):
        class Event:
            def __init__(self,a,b):self.a,self.b=a,b
            def name(self):return 'synthetic kernel'
            def is_user_annotation(self):return False
            def device_type(self):return 'CUDA'
            def start_ns(self):return self.a
            def end_ns(self):return self.b
        class Capture:
            def start(self):self.a=time.time_ns()
            def boundary(self):self.b=time.time_ns()
            def finish(self):return [Event(self.a,self.b)]
        monitor=self.monitor(profile_every=2,capture_factory=Capture)
        with monitor.step(1):pass
        with monitor.step(2):
            with monitor.phase('forward_host'):pass
        out=read(self.root/'run/profile.json')
        self.assertTrue(out['valid']);self.assertEqual(out['next_capture_update'],4)
        self.assertAlmostEqual(sum(out['exclusive_ms'].values()),out['step_ms'])
        self.assertTrue((self.root/'run/profiles/000000000002.json').is_file())
        sampler=Sampler(self.settings());sampler.collect()
        self.assertEqual(sampler.snapshot['runs'][0]['profile_history'][0]['update'],2)

    def test_durable_checkpoint_required_and_requests_are_deduplicated(self):
        monitor=self.monitor(heavy_every=2)
        with monitor.step(2):pass
        monitor.checkpoint(2,self.root/'missing.pt')
        self.assertFalse((self.root/'run/requests').exists())
        checkpoint=self.root/'checkpoint.pt';checkpoint.write_bytes(b'unit fixture; not model weights')
        monitor.checkpoint(2,checkpoint,terminal=True)
        monitor.checkpoint(2,checkpoint,terminal=True)
        requests=list((self.root/'run/requests').glob('*.json'))
        self.assertEqual(len(requests),1);self.assertTrue(read(requests[0])['terminal'])
        self.assertEqual(checkpoint.read_bytes(),b'unit fixture; not model weights')


class WorkerTests(Base):
    def request(self):
        monitor=self.monitor(heavy_every=2)
        with monitor.step(2):pass
        checkpoint=self.root/'checkpoint.pt';checkpoint.write_bytes(b'fixture')
        monitor.checkpoint(2,checkpoint)
        return next((self.root/'run/requests').glob('*.json'))

    def adapter(self, ready=True):
        calls=[]
        @contextmanager
        def capture_context(request):
            calls.append('acquire')
            try:yield CapturePlan(['python3','replay.py'],'GPU-abcdef','synthetic_kernel','unit test only')
            finally:calls.append('resume')
        return SimpleNamespace(ready=lambda r:ready,capture_context=capture_context),calls

    def test_not_ready_never_acquires_or_creates_receipt(self):
        path=self.request();adapter,calls=self.adapter(False)
        self.assertEqual(process_request(path,self.root/'run',adapter),'not_ready')
        self.assertEqual(calls,[]);self.assertFalse((self.root/'run/receipts').exists())

    def test_capture_failure_resumes_and_does_not_retry(self):
        path=self.request();adapter,calls=self.adapter()
        def fail(*args,**kwargs):raise TimeoutError('fixture timeout')
        with patch('mlmonitor.worker.shutil.which',return_value='/ncu'),patch('mlmonitor.worker.subprocess.run',return_value=SimpleNamespace(stdout='NCU fixture')):
            self.assertEqual(process_request(path,self.root/'run',adapter,minimum_free_bytes=0,runner=fail),'failed')
            self.assertEqual(process_request(path,self.root/'run',adapter,minimum_free_bytes=0,runner=fail),'already_attempted')
        self.assertEqual(calls,['acquire','resume'])
        self.assertEqual(read(self.root/'run/receipts'/path.name)['status'],'FAILED')

    def test_success_imports_real_format_and_releases_lease(self):
        path=self.request();adapter,calls=self.adapter()
        def runner(command,**kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-abcdef')
            self.assertIn('--launch-count',command)
            kwargs['output'].write_text('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","synthetic_kernel","gpu__time_duration.sum","nsecond","1000"\n')
        with patch('mlmonitor.worker.shutil.which',return_value='/ncu'),patch('mlmonitor.worker.subprocess.run',return_value=SimpleNamespace(stdout='NCU fixture')):
            self.assertEqual(process_request(path,self.root/'run',adapter,minimum_free_bytes=0,runner=runner),'complete')
        self.assertEqual(calls,['acquire','resume'])
        self.assertEqual(read(self.root/'run/nsight'/path.name)['kernels'][0]['duration_us'],1)

    def test_changed_checkpoint_is_not_replayed(self):
        path=self.request();adapter,calls=self.adapter()
        (self.root/'checkpoint.pt').write_bytes(b'changed fixture')
        with self.assertRaises(ValueError):process_request(path,self.root/'run',adapter)
        self.assertEqual(calls,[])

    def test_timed_out_replay_process_is_reaped(self):
        import os,sys
        output=self.root/'worker-output.txt'
        with self.assertRaises(TimeoutError):
            execute([sys.executable,'-u','-c','import os,time;print(os.getpid(),flush=True);time.sleep(30)'],
                    output=output,cwd=None,env={},timeout=.3)
        pid=int(output.read_text().strip())
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)


if __name__=='__main__':unittest.main()
