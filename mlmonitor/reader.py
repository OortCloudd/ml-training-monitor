"""Adapters preserve the working dashboard's histories and scalar boundaries."""
from collections import deque
from hashlib import sha256
import json
import os
import statistics

from .storage import finite, field, read


class JsonLines:
    def __init__(self):
        self.identity=None;self.anchor=None;self.offset=0;self.generation=0
        self.skipped=0;self.backlog=False;self.skipping=False

    def poll(self,path):
        if path is None:return []
        try:
            with path.open('rb') as handle:
                stat=os.fstat(handle.fileno());prefix=handle.readline(1_000_001)
                anchor=sha256(prefix).hexdigest() if prefix.endswith(b'\n') else None
                identity=(stat.st_dev,stat.st_ino)
                if identity!=self.identity or stat.st_size<self.offset or (self.anchor is not None and self.anchor!=anchor):
                    self.offset=0;self.generation+=1;self.skipped=0;self.skipping=False
                self.identity,self.anchor=identity,anchor
                handle.seek(self.offset);rows=[];consumed=0
                while consumed<8_000_000:
                    line=handle.readline(1_000_001)
                    if not line:break
                    consumed+=len(line)
                    if self.skipping or len(line)>1_000_000:
                        if not self.skipping:self.skipped+=1
                        self.skipping=not line.endswith(b'\n');self.offset=handle.tell();continue
                    if not line.endswith(b'\n'):break
                    self.offset=handle.tell()
                    try:row=json.loads(line)
                    except ValueError:self.skipped+=1;continue
                    if isinstance(row,dict):rows.append(row)
                    else:self.skipped+=1
                self.backlog=consumed>=8_000_000 and self.offset<stat.st_size
                return rows
        except OSError:return []


class RunReader:
    def __init__(self,spec):
        self.spec=spec;self.streams={key:JsonLines() for key in ('telemetry','health','profiles')}
        self.generations={};self.rows=deque(maxlen=1200)
        self.points=[];self.block=[];self.first=None;self.health=[];self.profiles=deque(maxlen=100)
        self.observations=deque(maxlen=601)

    def records(self,key):
        stream=self.streams[key];rows=stream.poll(self.spec['files'][key])
        reset=self.generations.get(key)!=stream.generation;self.generations[key]=stream.generation
        return rows,reset

    def read(self,now):
        source,reset=self.records('telemetry')
        if reset:self.rows.clear();self.points=[];self.block=[];self.first=None;self.observations.clear()
        mapping=self.spec['fields']
        for raw in source:
            row={k:finite(field(raw,path)) for k,path in mapping.items()}
            if row['step'] is None or row['step']<0 or row['step']!=int(row['step']):continue
            if self.rows and row['step']<=self.rows[-1]['step']:continue
            self.rows.append(row)
            if self.first is None:self.first=row
            self.block.append(row)
            if len(self.block)==100:self.points.append(self.mean_point(self.block));self.block=[]
        points=([self.first] if self.first else [])+self.points[:]
        if self.block and (not points or self.block[-1]['step']!=points[-1]['step']):points.append(self.mean_point(self.block))
        health,reset=self.records('health')
        if reset:self.health=[]
        for raw in health:
            fields={k:field(raw,v) for k,v in self.spec['diagnostic_fields'].items()}
            step=finite(fields.pop('step'))
            if step is None or step<0:continue
            healthy=fields.pop('healthy')
            if self.health and step<=self.health[-1]['step']:continue
            self.health.append({'step':step,'healthy':healthy if type(healthy) is bool else None,
                                **{k:finite(v) for k,v in fields.items()}})
        state=read(self.spec['files']['state']);binding=read(self.spec['files']['bindings'])
        latest=self.rows[-1] if self.rows else {}
        try:age=max(0,now-self.spec['files']['telemetry'].stat().st_mtime)
        except OSError:age=None
        if age is not None and latest.get('timestamp') is not None:age=max(0,now-latest['timestamp'])
        step=latest.get('step',0)
        status='Terminé' if state.get('status')=='COMPLETE' else ('Pas démarré' if not self.rows else
            'Télémétrie récente' if age is not None and age<120 else 'Télémétrie ancienne')
        if not self.streams['telemetry'].backlog:self.observations.append((now,step))
        while self.observations and now-self.observations[0][0]>600:self.observations.popleft()
        speed=None
        if self.observations and now-self.observations[0][0]>=60 and step>self.observations[0][1]:
            speed=(step-self.observations[0][1])/(now-self.observations[0][0])
        target=self.spec.get('target_updates',finite(state.get('target_updates')))
        eta=(target-step)/speed if speed and target and target>step and status=='Télémétrie récente' and not self.streams['telemetry'].backlog else None
        summary=self.health[-1] if self.health else {}
        means={k:self.mean([r.get(k) for r in list(self.rows)[-100:]]) for k in ('loss','seconds','gradient')}
        phase_values={}
        recent=list(self.rows)[-100:]
        for source,label in [('source_read_seconds','source_read'),('input_prepare_seconds','cpu_prepare'),
                             ('prepared_input_wait_seconds','input_wait')]:
            values=[r[source] for r in recent if r.get(source) is not None and r[source]>=0]
            if values:phase_values[label]={'seconds':sum(values),'calls':len(values),'steps':len(values)}
        logged_phases={'phases':phase_values,'steps':len(recent),'first_update':recent[0]['step'] if recent else None,
                       'last_update':step,'captured_at':now-age if age is not None else None} if phase_values and age is not None else None
        details=self.spec['details'] or binding.get('details',{})
        if not isinstance(details,dict):details={}
        details={str(k):str(v) for k,v in details.items() if isinstance(v,(str,int,float,bool)) and len(str(v))<=1000}
        fingerprint=self.spec.get('config_sha256');verified=not fingerprint or fingerprint==binding.get('config_sha256')
        return {'id':self.spec['id'],'name':self.spec['name'],'step_unit':self.spec['step_unit'],
            'status':status,'step':step,'target':target,'age':age,'eta':eta,**means,
            'checkpoint':finite(state.get('durable_checkpoint_update')),'health':summary.get('healthy'),
            'health_step':summary.get('step'),'health_points':self.health,'diagnostics':self.spec['diagnostics'],
            'network':{'status':'Configured metadata' if verified else 'Configuration mismatch',
                       'fields':list(details.items())} if details else None,
            'points':[{k:r.get(k) for k in ('step','loss','seconds','gradient')} for r in points],
            'window':len(self.rows),'catching_up':self.streams['telemetry'].backlog,
            'logged_phases':logged_phases,
            'observer_error':state.get('observer_error') if isinstance(state.get('observer_error'),str) else None}

    def profile_history(self,sha):
        rows,reset=self.records('profiles')
        if reset:self.profiles=deque(maxlen=100)
        for row in rows:
            if row.get('config_sha256')!=sha or not row.get('valid'):continue
            keys=('update','captured_at','step_ms','baseline_step_ms','capture_total_ms')
            point={k:finite(row.get(k)) for k in keys}
            if point['update'] is None or point['step_ms'] is None:continue
            self.profiles.append(point)
        return list(self.profiles)

    @staticmethod
    def mean(values):
        values=[v for v in values if v is not None]
        return statistics.mean(values) if values else None

    def mean_point(self,rows):
        return {'step':rows[-1]['step'],**{k:self.mean([r[k] for r in rows]) for k in ('loss','seconds','gradient')}}
