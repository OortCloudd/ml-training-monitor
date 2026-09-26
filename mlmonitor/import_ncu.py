#!/usr/bin/env python3
"""Import selected raw NCU CSV metrics; no inference from activity or fabricated roofline."""
import argparse,csv,json,re,sys
from hashlib import sha256
from pathlib import Path
from .gpu_efficiency import METRICS,sanitize_capture

def parse_value(value):
 s=str(value).strip()
 if not re.fullmatch(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?',s):return None
 return float(s.replace(',',''))

def import_rows(text,manifest):
 lines=text.splitlines();start=next((i for i,line in enumerate(lines) if '"Metric Name"' in line and '"Metric Value"' in line),None)
 if start is None:
  # NCU 2026 raw pages use one metric per column, with a separate units row.
  import io
  table=list(csv.reader(lines))
  start=next((i for i,row in enumerate(table) if 'Kernel Name' in row and 'gpu__time_duration.sum' in row),None)
  if start is None:raise ValueError('Expected NCU raw CSV with named metrics')
  headers,units=table[start],table[start+1]
  if len(headers)!=len(units):raise ValueError('NCU metric/unit columns differ')
  expanded=io.StringIO();writer=csv.DictWriter(expanded,fieldnames=['ID','Kernel Name','Metric Name','Metric Unit','Metric Value'],quoting=csv.QUOTE_ALL)
  writer.writeheader()
  for row in table[start+2:]:
   if len(row)!=len(headers):continue
   d=dict(zip(headers,row))
   for column,name in enumerate(headers):
    if name in {v[2] for v in METRICS.values()}|{'gpu__time_duration.sum'}:
     writer.writerow({'ID':d['ID'],'Kernel Name':d['Kernel Name'],'Metric Name':name,
                      'Metric Unit':units[column],'Metric Value':row[column]})
  return import_rows(expanded.getvalue(),manifest)
 mapping={v[2]:(k,v[1]) for k,v in METRICS.items()};kernels={}
 for row in csv.DictReader(lines[start:]):
  raw=row.get('Metric Name');ident=row.get('ID')
  if raw not in mapping and raw!='gpu__time_duration.sum':continue
  if not ident:continue
  kernel=kernels.setdefault(ident,{'label':row.get('Kernel Name') or 'Kernel '+ident,'metrics':{},'duration_us':None,'roofline':None})
  value=parse_value(row.get('Metric Value'));unit=row.get('Metric Unit','').strip()
  if raw=='gpu__time_duration.sum':
   factors={'nsecond':.001,'usecond':1,'msecond':1000,'second':1e6,'ns':.001,'us':1,'ms':1000,'s':1e6}
   kernel['duration_us']=value*factors[unit] if value is not None and unit in factors else None
  else:
   name,expected=mapping[raw]
   # NCU 2026 prints `warp` for the exact `.per_cycle_active` eligible-warps metric;
   # the denominator is part of the metric identity, no numeric rescaling.
   accepted=unit in ('%','percent') if expected=='%' else unit in ('warp/cycle','warps/cycle','warp')
   if not accepted:raise ValueError('Unexpected unit for '+raw+': '+unit)
   if name in kernel['metrics']:raise ValueError('Duplicate metric/kernel ID; split the export by capture/process')
   kernel['metrics'][name]=value
 capture={**manifest,'schema':'gpu-nsight-capture-v1','kernels':list(kernels.values())}
 return sanitize_capture(capture)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Refusing to overwrite an existing capture')
 payload=a.csv.read_bytes();result=import_rows(payload.decode('utf-8-sig'),json.loads(a.manifest.read_text()))
 result['csv_sha256']=sha256(payload).hexdigest()
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print('Imported',len(result['kernels']),'kernels; absent counters/roofline remain unavailable')
if __name__=='__main__':main()
