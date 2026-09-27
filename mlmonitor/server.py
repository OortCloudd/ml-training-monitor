"""Training monitoring with opt-in proposal review; no training-command endpoint."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hmac
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from .config import load_config
from .dmon_monitor import DmonMonitor
from .gpu_efficiency import capabilities, load_captures, METRICS, LIVE_METRICS, window_summary
from .hardware import Hardware
from .reader import RunReader
from .storage import read, finite
from .step_attribution import load_capture
from .recorder import next_capture
from .advice import ProposalStore, opportunities

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
        support=settings.get('decision_support', {'enabled':False})
        self.proposals=ProposalStore(support['directory']) if support['enabled'] else None
        self.review_token=secrets.token_urlsafe(32) if self.proposals else None

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
                                         'requested_interval_seconds':self.settings['interval']},
                           'decision_support':{'enabled':self.proposals is not None}}

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
        def send_json(self, value, status=200):
            body=json.dumps(value,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path=='/':
                text=(BASE/'index.html').read_text().replace('__MLMONITOR_REVIEW_TOKEN__',json.dumps(sampler.review_token))
                body=text.encode();kind='text/html; charset=utf-8'
            elif path=='/api/metrics':
                with sampler.lock:body=json.dumps(sampler.snapshot,allow_nan=False).encode()
                kind='application/json'
            elif path=='/api/decisions':
                if sampler.proposals is None:self.send_json({'enabled':False});return
                with sampler.lock:snapshot=sampler.snapshot
                self.send_json({'enabled':True,'opportunities':opportunities(snapshot),**sampler.proposals.listing(snapshot)})
                return
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

        def do_POST(self):
            # The only write route records a human review. No command execution.
            path=self.path.split('?',1)[0]
            if path!='/api/proposals/review':self.send_json({'error':'Not found'},404);return
            if sampler.proposals is None:self.send_json({'error':'Decision support is disabled'},403);return
            port=self.server.server_address[1]
            bind=self.server.server_address[0]
            hosts={bind} if bind not in ('0.0.0.0','::') else set()
            hosts.update(('127.0.0.1','localhost','::1'))
            try:
                origin=urlsplit(self.headers.get('Origin',''))
                host=urlsplit('http://'+self.headers.get('Host',''))
                same_origin=(origin.scheme=='http' and origin.hostname in hosts and origin.port==port and
                             host.hostname==origin.hostname and host.port==port)
            except ValueError:same_origin=False
            token=self.headers.get('X-Review-Token','')
            if not same_origin or not hmac.compare_digest(token.encode('utf-8'),sampler.review_token.encode('ascii')):
                self.send_json({'error':'Review requires the dashboard page on the configured host; reload it if the server restarted.'},403);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16000 or self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    raise ValueError('Review must be a small JSON request')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict) or set(payload)-{'id','content_sha256','decision','comment'}:
                    raise ValueError('Invalid review fields')
                with sampler.lock:snapshot=sampler.snapshot
                result=sampler.proposals.review(payload.get('id'),payload.get('content_sha256'),
                                               payload.get('decision'),payload.get('comment',''),snapshot)
            except (ValueError,TypeError,KeyError) as exc:
                self.send_json({'error':str(exc)},409);return
            self.send_json(result)

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
