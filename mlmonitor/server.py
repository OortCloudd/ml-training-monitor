"""Read-only dashboard retaining the existing monitoring panels and collectors."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

from .config import load_config
from .dmon_monitor import DmonMonitor
from .gpu_efficiency import capabilities, load_captures, METRICS, LIVE_METRICS, window_summary
from .hardware import Hardware
from .reader import RunReader
from .storage import read, finite
from .step_attribution import load_capture
from .recorder import next_capture

BASE = Path(__file__).resolve().parent


class Sampler:
    def __init__(self, settings):
        self.settings = settings
        self.hardware = Hardware(settings)
        self.readers = [RunReader(r) for r in settings['runs']]
        self.history = deque(maxlen=max(60, 3600 // settings['interval']))
        self.dmon = DmonMonitor(settings['gpu_indices'], settings['gpu_enabled'])
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.snapshot = {'loading': True}

    def collect(self):
        now = time.time()
        started=time.monotonic()
        hardware = self.hardware.read()
        hardware_finished=time.time()
        runs, captures, errors = [], [], []
        for reader in self.readers:
            spec = reader.spec
            run_now=time.time()
            run = reader.read(run_now)
            binding = read(spec['files']['bindings'])
            sha = spec.get('config_sha256') or binding.get('config_sha256')
            run['config_sha256']=sha
            profile_path = spec['files']['profile']
            run['bottleneck'] = load_capture(profile_path, sha, None, run_now, completed=run['status']=='Terminé',
                                              current_step=run['step']) if profile_path and sha else {'status':'disabled'}
            logged_phases=run.pop('logged_phases')
            run['light_profile'] = validated_auxiliary(spec['files']['light_profile'], sha, 'phases') or logged_phases
            run['operations'] = validated_auxiliary(spec['files']['operations'], sha, 'operations')
            run['profile_history'] = reader.profile_history(sha)
            every = spec.get('capture_every', binding.get('profile_every'))
            steps = spec.get('capture_steps', binding.get('profile_steps', []))
            run['next_profile_update'] = next_capture(run['step'], every, steps)
            if not every and not steps and run['bottleneck']['status']=='waiting':
                run['bottleneck']={'status':'disabled'}
            run['profile_enabled'] = bool(every or steps)
            run['capture_every'] = every
            run['heavy_enabled'] = bool(binding.get('heavy_every') or binding.get('heavy_steps'))
            folder = spec['files']['captures']
            if folder:
                found, errs = load_captures(folder, {run['id']: sha})
                captures.extend(c for c in found if c['run_id']==run['id'])
                errors.extend(errs)
            runs.append(run)
        for gpu in hardware['gpus']:
            gpu['run_id']=''
            gpu['run_name']='Charge non identifiée'
        self.history.append({'time':hardware_finished,'cpu':hardware['cpu'],'gpus':[
            {k:g.get(k) for k in ('index','uuid','utilization','memory_activity','power','temperature',
                                'sm_clock','memory_clock','memory_used','power_cap','thermal_cap')}
            for g in hardware['gpus']]})
        published_at=time.time()
        cap=capabilities();cap['live_interval_seconds']=self.settings['interval']
        performance={'capabilities':cap,'dmon_live':self.dmon.snapshot(published_at),
            'windows':{str(g['index']):window_summary(self.history,g['index'],published_at,interval=self.settings['interval'])
                       for g in hardware['gpus']},
            'captures':sorted(captures,key=lambda c:c['timestamp'],reverse=True),'errors':errors,
            'metric_definitions':{k:{'label':v[0],'unit':v[1],'raw_name':v[2]} for k,v in METRICS.items()},
            'live_metric_definitions':LIVE_METRICS,
            'dmon_capture':{}}
        with self.lock:
            self.snapshot={'schema':'ml-monitor-snapshot-v1','timestamp':hardware_finished,'title':self.settings['title'],
                           'hardware':hardware,'runs':runs,'performance':performance,
                           'history':list(self.history),'interval':self.settings['interval'],
                           'collection':{'started_at':now,'hardware_read_finished_at':hardware_finished,
                                         'published_at':published_at,'duration_seconds':time.monotonic()-started,
                                         'requested_interval_seconds':self.settings['interval']}}

    def loop(self):
        while not self.stop.is_set():
            start=time.monotonic()
            try:self.collect()
            except Exception as exc:print('Sampling error:',type(exc).__name__,flush=True)
            self.stop.wait(max(.1,self.settings['interval']-(time.monotonic()-start)))

    def close(self):
        self.stop.set()
        self.dmon.close()


def validated_auxiliary(path, sha, field):
    raw=read(path)
    if not sha or raw.get('config_sha256')!=sha or finite(raw.get('captured_at')) is None:return None
    items=raw.get(field)
    if not isinstance(items,dict):return None
    clean={}
    for key,value in items.items():
        if not isinstance(key,str) or len(key)>64 or not isinstance(value,dict):return None
        seconds,calls=finite(value.get('seconds')),value.get('calls')
        if seconds is None or seconds<0 or type(calls) is not int or calls<0:return None
        clean[key]={'seconds':seconds,'calls':calls,'steps':finite(value.get('steps'))}
    return {'captured_at':raw['captured_at'],'source':'monitor_summary',
            'scope':raw.get('scope') if isinstance(raw.get('scope'),str) else None,field:clean,**{
        k:finite(raw.get(k)) for k in ('window_seconds','first_update','last_update','steps','step_seconds')}}


def handler(sampler):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path=='/':body=(BASE/'index.html').read_bytes();kind='text/html; charset=utf-8'
            elif path=='/api/metrics':
                with sampler.lock:body=json.dumps(sampler.snapshot,allow_nan=False).encode()
                kind='application/json'
            elif path=='/api/dmon/raw':body=sampler.dmon.snapshot()['raw'].encode();kind='text/plain; charset=utf-8'
            elif path=='/favicon.ico':self.send_response(204);self.end_headers();return
            else:self.send_error(404);return
            self.send_response(200)
            for k,v in [('Content-Type',kind),('Content-Length',str(len(body))),('Cache-Control','no-store'),
                        ('X-Content-Type-Options','nosniff'),('Content-Security-Policy',"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'")]:
                self.send_header(k,v)
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*args):pass
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8790)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    try:settings=load_config(args.config)
    except (OSError,ValueError,KeyError,TypeError) as exc:parser.error(str(exc))
    if args.check:print('Configuration valid:',len(settings['runs']),'runs');return
    sampler=Sampler(settings)
    server=ThreadingHTTPServer((args.host,args.port),handler(sampler))
    threads=[threading.Thread(target=fn,daemon=True) for fn in (sampler.loop,sampler.dmon.loop)]
    for thread in threads:thread.start()
    print(f'Monitor listening on http://{args.host}:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        sampler.close();server.server_close()
        for thread in threads:thread.join(timeout=5)


if __name__=='__main__':main()
