"""Unprivileged continuous dmon with an in-memory two-minute raw window."""
from collections import deque
from datetime import datetime
import math,shutil,subprocess,threading,time
from .gpu_efficiency import stats

class DmonMonitor:
    def __init__(self, indices=None, enabled=True):
        self.indices=indices;self.enabled=enabled;self.stop=threading.Event()
        self.lock=threading.Lock();self.rows=deque(maxlen=20000);self.headers=[];self.columns=None
        self.error=None;self.process=None

    def accept(self,line):
        line=line.rstrip('\r\n')
        if line.startswith('#Date'):
            columns=line[1:].split()
            if not {'Date','Time','gpu','sm','mem','pwr','pclk','mclk','fb'}.issubset(columns):return
            with self.lock:self.columns=columns;self.headers=[line]
            return
        if line.startswith('#YYYYMMDD'):
            with self.lock:
                if self.headers:self.headers=self.headers[:1]+[line]
            return
        if line.startswith('#') or not self.columns:return
        values=line.split()
        if len(values)!=len(self.columns):return
        fields=dict(zip(self.columns,values))
        try:
            stamp=datetime.strptime(fields.pop('Date')+' '+fields.pop('Time'),'%Y%m%d %H:%M:%S').astimezone().timestamp()
            point={k:None if v=='-' else float(v) for k,v in fields.items()}
            if any(v is not None and not math.isfinite(v) for v in point.values()):return
            if self.indices is not None and point['gpu'] not in self.indices:return
            point.update(time=stamp,raw=line)
        except (ValueError,KeyError):return
        with self.lock:self.rows.append(point);self.error=None

    def snapshot(self,now=None):
        now=time.time() if now is None else now
        with self.lock:
            rows=[dict(r) for r in self.rows if now-120<r['time']<=now+1]
            headers=list(self.headers);error=self.error
            known_indices={int(r['gpu']) for r in self.rows}
        groups=[]
        for index in sorted(set(self.indices or [])|known_indices):
            selected=[r for r in rows if r['gpu']==index];gaps=[b['time']-a['time'] for a,b in zip(selected,selected[1:])]
            groups.append({'index':index,'samples':len(selected),'span_seconds':selected[-1]['time']-selected[0]['time'] if len(selected)>1 else 0,
                'max_gap_seconds':max(gaps) if gaps else None,'has_gaps':any(x>2.5 for x in gaps),
                'stats':{k:stats([r.get(k) for r in selected]) for k in ['sm','mem','pwr','gtemp','pclk','mclk','fb']},
                'points':[{k:v for k,v in r.items() if k!='raw'} for r in selected]})
        last=max((r['time'] for r in rows),default=None)
        return {'schema':'gpu-dmon-live-v1','command':' '.join(self.command()),
                'window_seconds':120,'last_sample_at':last,'age_seconds':now-last if last is not None else None,
                'status':'waiting' if last is None else 'stale' if now-last>5 else 'collecting',
                'error':error,'gpus':groups,'raw':'\n'.join(headers+[r['raw'] for r in rows])+('\n' if headers else '')}

    def loop(self):
        if not self.enabled or self.indices==[]:return
        command=self.command()
        if shutil.which('stdbuf'):command=['stdbuf','-oL',*command]
        while not self.stop.is_set():
            try:
                self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                if self.stop.is_set():self.process.terminate()
                for line in self.process.stdout:
                    if self.stop.is_set():
                        self.process.terminate();break
                    self.accept(line)
                code=self.process.wait()
                with self.lock:self.error=f'Collecteur dmon interrompu (code {code}), reprise automatique.'
            except OSError:
                with self.lock:self.error='Collecteur dmon indisponible, nouvelle tentative automatique.'
            finally:self.process=None
            self.stop.wait(3)

    def command(self):
        devices=['-i',','.join(map(str,self.indices))] if self.indices else []
        return ['nvidia-smi','dmon',*devices,'-s','pucm','-d','1','-o','DT']

    def close(self):
        self.stop.set()
        process=self.process
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:process.kill();process.wait()
