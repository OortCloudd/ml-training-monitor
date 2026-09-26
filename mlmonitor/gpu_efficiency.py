"""Temporal NVML summaries and explicitly dated, sanitized Nsight captures."""
from datetime import datetime
import json,math,re,shutil
from pathlib import Path

METRICS={
 'sm_throughput_pct':('SOL calcul SM','%', 'sm__throughput.avg.pct_of_peak_sustained_elapsed'),
 'memory_throughput_pct':('SOL mémoire (agrégat)','%', 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'),
 'dram_throughput_pct':('Débit DRAM / pic soutenu','%', 'dram__throughput.avg.pct_of_peak_sustained_elapsed'),
 'l2_throughput_pct':('Débit L2 / pic soutenu','%', 'lts__throughput.avg.pct_of_peak_sustained_elapsed'),
 'tensor_active_pct':('Activité du pipeline Tensor','%', 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),
 'occupancy_pct':('Occupancy atteinte','%', 'sm__warps_active.avg.pct_of_peak_sustained_active'),
 'eligible_warps':('Warps éligibles / ordonnanceur','warps/cycle', 'smsp__warps_eligible.avg.per_cycle_active'),
 'issue_active_pct':('Cycles avec émission','%', 'smsp__issue_active.avg.pct_of_peak_sustained_active'),
}
LIVE_KEYS=('utilization','memory_activity','power','temperature','sm_clock','memory_clock','memory_used')

def finite(v):
 return v if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) else None

def stats(values):
 a=sorted(x for x in values if finite(x) is not None)
 if not a:return {'n':0,'mean':None,'p50':None,'p95':None,'min':None,'max':None}
 def q(p):
  i=(len(a)-1)*p;j=int(i);return a[j]+(a[min(j+1,len(a)-1)]-a[j])*(i-j)
 return {'n':len(a),'mean':sum(a)/len(a),'p50':q(.5),'p95':q(.95),'min':a[0],'max':a[-1]}

def window_summary(history,index,now,seconds=300,interval=1):
 rows=[]
 for sample in history:
  if now-seconds<=sample['time']<=now:
   gpu=next((g for g in sample['gpus'] if g['index']==index),None)
   if gpu is not None:rows.append((sample['time'],gpu))
 gaps=[b[0]-a[0] for a,b in zip(rows,rows[1:])]
 return {'requested_seconds':seconds,'samples':len(rows),
  'start':rows[0][0] if rows else None,'end':rows[-1][0] if rows else None,
  'span_seconds':rows[-1][0]-rows[0][0] if len(rows)>1 else 0,
  'max_gap_seconds':max(gaps) if gaps else None,
  'median_interval_seconds':stats(gaps)['p50'],
  'has_gaps':any(g>2.5*interval for g in gaps),
  'stats':{k:stats([g.get(k) for _,g in rows]) for k in LIVE_KEYS},
  'power_cap_observations':sum(g.get('power_cap') is True for _,g in rows),
  'power_cap_samples':sum(isinstance(g.get('power_cap'),bool) for _,g in rows)}

def capabilities():
 restricted=None
 try:
  m=re.search(r'^RmProfilingAdminOnly:\s*(\d+)',Path('/proc/driver/nvidia/params').read_text(),re.M)
  if m:restricted=bool(int(m.group(1)))
 except OSError:pass
 return {'ncu_available':shutil.which('ncu') is not None,'admin_only_counters':restricted,
  'gpm_note':'',
  'source':'nvidia-smi / NVML en direct ; Nsight Compute uniquement via captures importées',
  'live_interval_seconds':1,'rolling_window_seconds':300}

def sanitize_capture(raw):
 if raw.get('schema')!='gpu-nsight-capture-v1':raise ValueError('unknown capture schema')
 if not isinstance(raw.get('run_id'),str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',raw['run_id']):raise ValueError('invalid run')
 stamp=datetime.fromisoformat(raw['captured_at'].replace('Z','+00:00'))
 if stamp.tzinfo is None:raise ValueError('capture timezone required')
 if not re.fullmatch(r'GPU-[a-fA-F0-9-]+',raw['gpu_uuid']):raise ValueError('GPU identity required')
 if not re.fullmatch(r'[a-fA-F0-9]{64}',raw['config_sha256']):raise ValueError('configuration digest required')
 if not isinstance(raw.get('tool_version'),str) or not raw['tool_version']:raise ValueError('tool version required')
 if not isinstance(raw.get('scope'),str) or not raw['scope']:raise ValueError('capture scope required')
 kernels=[]
 for k in raw['kernels'][:64]:
  metrics={}
  for key,(_,unit,_) in METRICS.items():
   v=finite(k.get('metrics',{}).get(key))
   if v is not None and (v<0 or (unit=='%' and v>100.5)):v=None
   metrics[key]=v
  roof=k.get('roofline');clean_roof=None
  if isinstance(roof,dict):
   keys=['arithmetic_intensity','achieved_tflops','compute_ceiling_tflops','bandwidth_tb_s']
   if (roof.get('precision') in ['FP64','FP32','FP16','BF16','FP8'] and roof.get('memory_level') in ['DRAM','L2','L1']
       and isinstance(roof.get('operation_convention'),str) and roof['operation_convention']
       and all(finite(roof.get(n)) is not None and roof[n]>0 for n in keys)):
    clean_roof={n:roof[n] for n in keys+['precision','memory_level','operation_convention']}
  duration=finite(k.get('duration_us'))
  kernels.append({'label':str(k.get('label','Kernel'))[:160],'duration_us':duration if duration is not None and duration>=0 else None,
                  'metrics':metrics,'roofline':clean_roof})
 if not kernels:raise ValueError('empty capture')
 return {'schema':raw['schema'],'run_id':raw['run_id'],'captured_at':raw['captured_at'],'timestamp':stamp.timestamp(),
         'gpu_uuid':raw['gpu_uuid'],'config_sha256':raw['config_sha256'],
         'tool_version':raw['tool_version'][:80],'scope':raw['scope'][:300],
         'checkpoint_update':finite(raw.get('checkpoint_update')),'kernels':kernels,
         'profile_overhead_pct':finite(raw.get('profile_overhead_pct'))}

def load_captures(folder,bindings):
 captures=[];errors=[]
 for path in sorted(folder.glob('*.json'), key=lambda p:p.stat().st_mtime, reverse=True)[:32]:
  try:
   if path.stat().st_size>2_000_000:raise ValueError('capture summary too large')
   c=sanitize_capture(json.loads(path.read_text()))
   c['matches_current_config']=bindings.get(c['run_id'])==c['config_sha256']
   captures.append(c)
  except (OSError,ValueError,KeyError,TypeError,AttributeError):errors.append('Une capture Nsight invalide a été ignorée.')
 return sorted(captures,key=lambda c:c['timestamp'],reverse=True),errors
