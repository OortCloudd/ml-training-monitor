import csv,io,json,tempfile,unittest
from pathlib import Path
import sys
from mlmonitor.gpu_efficiency import stats,window_summary,sanitize_capture,load_captures,METRICS
sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'))
from mlmonitor.import_ncu import import_rows

class EfficiencyTests(unittest.TestCase):
 def test_missing_is_not_zero_and_quantiles(self):
  s=stats([None,float('nan'),0,10,20,30,40]);self.assertEqual(s['n'],5);self.assertEqual(s['mean'],20);self.assertEqual(s['p50'],20);self.assertEqual(s['p95'],38)
  self.assertIsNone(stats([None])['mean'])
 def test_window_selection_and_gaps(self):
  h=[{'time':t,'gpus':[{'index':0,'utilization':v,'memory_activity':None,'power_cap':False}]} for t,v in [(1,100),(301,20),(302,40),(310,60)]]
  s=window_summary(h,0,310,seconds=20)
  self.assertEqual(s['samples'],3);self.assertEqual(s['span_seconds'],9);self.assertEqual(s['stats']['utilization']['mean'],40);self.assertTrue(s['has_gaps']);self.assertIsNone(s['stats']['memory_activity']['mean']);self.assertEqual(s['power_cap_observations'],0)
 def manifest(self):
  return {'schema':'gpu-nsight-capture-v1','run_id':'example_run','captured_at':'2026-09-24T12:00:00+00:00','gpu_uuid':'GPU-abcdef-1234','config_sha256':'a'*64,'tool_version':'test fixture','scope':'synthetic unit test, not production','checkpoint_update':77045}
 def test_capture_whitelist_and_missing_roofline(self):
  r={**self.manifest(),'private_field':'must not leave file','kernels':[{'label':'gemm','metrics':{'sm_throughput_pct':80,'occupancy_pct':None,'tensor_active_pct':999},'roofline':{'precision':'BF16','arithmetic_intensity':100}}]}
  out=sanitize_capture(r)
  self.assertNotIn('private_field',out);self.assertEqual(out['kernels'][0]['metrics']['sm_throughput_pct'],80);self.assertIsNone(out['kernels'][0]['metrics']['tensor_active_pct']);self.assertIsNone(out['kernels'][0]['metrics']['occupancy_pct']);self.assertIsNone(out['kernels'][0]['roofline'])
 def test_roofline_requires_precision_memory_level_and_convention(self):
  roof={'precision':'BF16','memory_level':'DRAM','operation_convention':'FMA=2; dense BF16 fixture','arithmetic_intensity':20,'achieved_tflops':10,'compute_ceiling_tflops':100,'bandwidth_tb_s':1}
  r={**self.manifest(),'kernels':[{'label':'test','metrics':{},'roofline':roof}]}
  self.assertEqual(sanitize_capture(r)['kernels'][0]['roofline']['precision'],'BF16')
  del roof['operation_convention'];self.assertIsNone(sanitize_capture(r)['kernels'][0]['roofline'])
 def test_import_exact_names_units_and_no_estimated_counters(self):
  f=io.StringIO();w=csv.writer(f,quoting=csv.QUOTE_ALL)
  w.writerow(['ID','Kernel Name','Metric Name','Metric Unit','Metric Value'])
  w.writerow(['0','gemm',METRICS['sm_throughput_pct'][2],'%','42.5'])
  w.writerow(['0','gemm','gpu__time_duration.sum','nsecond','1,000'])
  w.writerow(['0','gemm','unrecognized_counter','%','99'])
  out=import_rows('==PROF== metadata\n'+f.getvalue(),self.manifest());k=out['kernels'][0]
  self.assertEqual(k['duration_us'],1);self.assertEqual(k['metrics']['sm_throughput_pct'],42.5);self.assertIsNone(k['metrics']['dram_throughput_pct']);self.assertIsNone(k['roofline'])
  with self.assertRaises(ValueError):import_rows(f.getvalue().replace('"%"','"byte"'),self.manifest())
 def test_capture_config_match_and_bad_input(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'one.json').write_text(json.dumps({**self.manifest(),'kernels':[{'label':'test','metrics':{}}]}));(p/'bad.json').write_text('{}')
   captures,errors=load_captures(p,{'example_run':'b'*64});self.assertFalse(captures[0]['matches_current_config']);self.assertEqual(len(errors),1)
   captures,_=load_captures(p,{'example_run':'a'*64});self.assertTrue(captures[0]['matches_current_config'])


class DmonTests(unittest.TestCase):
 def test_continuous_raw_window_expires_and_preserves_missing(self):
  from mlmonitor.dmon_monitor import DmonMonitor
  from datetime import datetime,timedelta
  d=DmonMonitor();d.accept('#Date Time gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk fb bar1 ccpm')
  d.accept('#YYYYMMDD HH:MM:SS Idx W C C % % % % % % MHz MHz MB MB MB')
  start=datetime(2026,9,24,16,0,0)
  for i in range(130):
   for gpu in (0,1):d.accept((start+timedelta(seconds=i)).strftime('%Y%m%d %H:%M:%S')+f' {gpu} 200 50 - 90 40 0 0 0 0 9001 2200 16000 16001 0')
  s=d.snapshot(start.astimezone().timestamp()+129.4)
  self.assertEqual([g['samples'] for g in s['gpus']],[120,120]);self.assertEqual(s['status'],'collecting');self.assertEqual(len(s['raw'].splitlines()),242)
  self.assertNotIn('20260924 16:00:00',s['raw']);self.assertIn('20260924 16:02:09',s['raw']);self.assertIsNone(s['gpus'][0]['points'][0]['mtemp'])
  self.assertNotIn('raw',s['gpus'][0]['points'][0]);self.assertEqual(s['gpus'][0]['stats']['sm']['mean'],90)
  self.assertEqual(d.snapshot(start.astimezone().timestamp()+400)['gpus'][0]['samples'],0)
 def test_bad_row_and_stale_samples(self):
  from mlmonitor.dmon_monitor import DmonMonitor
  from datetime import datetime
  d=DmonMonitor();d.accept('unexpected output');self.assertEqual(d.snapshot()['status'],'waiting')
  d.accept('#Date Time gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk fb bar1 ccpm')
  d.accept('20260924 16:00:00 0 200 50 - nan 40 0 0 0 0 9001 2200 16000 16001 0');self.assertEqual(len(d.rows),0)
  d.accept('20260924 16:00:00 0 200 50 - 90 40 0 0 0 0 9001 2200 16000 16001 0')
  self.assertEqual(d.snapshot(datetime(2026,9,24,16,0,20).astimezone().timestamp())['status'],'stale')

if __name__=='__main__':unittest.main()
