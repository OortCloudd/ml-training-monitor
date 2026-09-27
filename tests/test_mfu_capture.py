import json
from pathlib import Path
import tempfile
import unittest
from mlmonitor.mfu import validate_contract
from mlmonitor.mfu_capture import summarize_capture,reduce_data_parallel_step
from mlmonitor.uncertainty import rate_interval
from mlmonitor.config import load_config
from mlmonitor.reader import RunReader

CFG='a'*64;EST='b'*64

def contract():
 return validate_contract(dict(config_sha256=CFG,gpu_model='NVIDIA L40S',gpu_count=1,precision='bf16',
     sparsity='dense',peak_tflops_per_gpu=100,peak_source='Synthetic test ceiling',flop_source='Synthetic fixture',
     timing_scope='unprofiled real updates',mode='capture',gpu_uuids=['GPU-test'],estimator_sha256=EST))

def capture(n=100):
 rows=[dict(valid=True,profiled=False,update=i,model_flops=20e12,executed_model_flops=30e12,
            update_seconds=1.,config_sha256=CFG,estimator_sha256=EST,gpu_uuids=['GPU-test'],
            telemetry_record_sha256='c'*64) for i in range(1,n+1)]
 return dict(schema='mfu-periodic-capture-v1',run_id='test',valid=True,config_sha256=CFG,estimator_sha256=EST,
             gpu_uuids=['GPU-test'],gpu_model='NVIDIA L40S',precision='bf16',sparsity='dense',
             profile_timing_used_for_mfu=False,dispatcher_flops=30e12,
             operator_flops={'synthetic':30e12},flops_by_dtype={'torch.bfloat16':30e12},
             work={'update':n+1,'config_sha256':CFG,'model_flops':20e12,'executed_model_flops':30e12,'recompute_flops':10e12,'update_seconds':99},
             captured_at=1000,milestone=n+1,baseline_steps=rows,
             arithmetic_scope='synthetic matrices',systematic_limit='fixture',nsight_join={'run_id':'test','config_sha256':CFG,'milestone':n+1})

class CaptureTests(unittest.TestCase):
 def read(self,data,now=1100,run_id='test',binding=None):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'capture.json';p.write_text(json.dumps(data))
   return summarize_capture(p,contract(),binding or {'config_sha256':CFG},now,run_id=run_id)
 def test_profile_overhead_never_enters_the_denominator(self):
  r=self.read(capture());self.assertEqual(r['percent'],20)
  self.assertEqual(r['window_seconds'],100);self.assertEqual(r['capture_update'],101)
  self.assertEqual(r['capture_age_seconds'],100);self.assertFalse(r['current_live_estimate'])
  self.assertEqual(r['confidence_interval']['lower_percent'],20)
  self.assertEqual(self.read(capture())['confidence_interval']['lower_percent'],20)
 def test_count_identity_precision_and_allocation_fail_closed(self):
  for key,value in [('valid',False),('estimator_sha256','x'),('gpu_uuids',['other']),('dispatcher_flops',1),
                    ('precision','fp32'),('profile_timing_used_for_mfu',True)]:
   c=capture();c[key]=value;self.assertIsNone(self.read(c)['percent'],key)
 def test_capture_must_belong_to_expected_run_not_only_same_configuration(self):
  c=capture();c['run_id']='another-run';c['nsight_join']['run_id']='another-run'
  self.assertEqual(self.read(c)['reason'],'capture_run_mismatch')
  self.assertEqual(self.read(capture(),run_id=None)['reason'],'capture_run_mismatch')
  self.assertEqual(self.read(capture(),run_id='display-alias',binding={'config_sha256':CFG,'run_id':'test'})['percent'],20)
 def test_reader_binds_capture_to_native_run_or_registered_id(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder)
   (root/'training_telemetry.jsonl').write_text(json.dumps({'update':101})+'\n')
   (root/'run_bindings.json').write_text(json.dumps({'config_sha256':CFG}))
   (root/'capture.json').write_text(json.dumps(capture()))
   config_path=root/'monitor.json'
   config_path.write_text(json.dumps({'gpu_enabled':False,'runs':[{'id':'test','directory':'.',
       'mfu':contract(),'files':{'mfu_capture':'capture.json'}}]}))
   reader=RunReader(load_config(config_path)['runs'][0])
   self.assertEqual(reader.read(1100)['mfu']['percent'],20)
   c=capture();c['run_id']='other';c['nsight_join']['run_id']='other'
   (root/'capture.json').write_text(json.dumps(c))
   self.assertEqual(reader.read(1100)['mfu']['reason'],'capture_run_mismatch')
   (root/'run_bindings.json').write_text(json.dumps({'config_sha256':CFG,'run_id':'other'}))
   self.assertEqual(reader.read(1100)['mfu']['percent'],20)
 def test_a_profiled_reference_or_gap_is_rejected(self):
  c=capture();c['baseline_steps'][20]['profiled']=True;self.assertIsNone(self.read(c)['percent'])
  c=capture();c['baseline_steps'].pop(20);self.assertEqual(self.read(c)['reason'],'capture_window_gap')
 def test_early_small_window_has_no_invented_confidence_interval(self):
  c=capture(49);r=self.read(c)
  self.assertEqual(r['percent'],20);self.assertEqual(r['confidence_interval']['status'],'insufficient_window')
  self.assertEqual(r['observed_range_kind'],'descriptive_not_confidence')
 def test_nonstationary_window_has_no_confident_extrapolation(self):
  c=capture()
  for row in c['baseline_steps'][50:]:row['update_seconds']=2.
  self.assertEqual(self.read(c)['confidence_interval']['status'],'nonstationary')
 def test_ddp_uses_single_coordinator_time_and_unique_local_work(self):
  rows=[dict(run_id='r',config_sha256=CFG,step=8,rank=i,gpu_uuid=f'g{i}',world_size=2,
       work_scope='rank_local_unique_samples',model_flops=100e12,seconds=i+1) for i in range(2)]
  args=dict(run_id='r',config_sha256=CFG,step=8,gpu_uuids=['g0','g1'],step_seconds=2)
  r=reduce_data_parallel_step(rows,**args);self.assertEqual(r['seconds'],2);self.assertEqual(r['model_flops'],200e12)
  for bad in ([rows[0]],[rows[0],rows[0]],[rows[0],dict(rows[1],run_id='independent')],
              [rows[0],dict(rows[1],work_scope='global')],[rows[0],dict(rows[1],step=9)]):
   with self.assertRaises(ValueError):reduce_data_parallel_step(bad,**args)

if __name__=='__main__':unittest.main()
