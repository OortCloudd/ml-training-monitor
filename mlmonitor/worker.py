"""Optional checkpoint replay worker. Never started by the web server.

The worker supplies deduplication, capture execution, time/storage bounds, import,
and receipts. A user-owned adapter supplies readiness and an acquisition/recovery
context for their particular training system.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import importlib.util
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from .gpu_efficiency import METRICS
from .import_ncu import import_rows
from .storage import atomic, read


@dataclass
class CapturePlan:
    command: list[str]
    gpu_uuid: str
    kernel_regex: str
    scope: str
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    launch_count: int = 4
    launch_skip: int = 0


def validate_request(path, directory):
    request = read(path)
    if request.get('schema') != 'ml-monitor-capture-request-v1':
        raise ValueError('unknown request schema')
    binding = read(directory / 'run_bindings.json')
    if request.get('run_id') != binding.get('run_id') or request.get('config_sha256') != binding.get('config_sha256'):
        raise ValueError('request is for a different run/configuration')
    update = request.get('update')
    if type(update) is not int or update < 1:
        raise ValueError('invalid update')
    key = f"{request['run_id']}-{request['config_sha256'][:16]}-{update:012d}"
    if request.get('key') != key or path.stem != key:
        raise ValueError('request identity mismatch')
    checkpoint = Path(request['checkpoint'])
    if not checkpoint.is_file():
        return None
    stat = checkpoint.stat()
    if stat.st_size != request.get('checkpoint_size') or stat.st_mtime_ns != request.get('checkpoint_mtime_ns'):
        raise ValueError('checkpoint changed since the request was created')
    return request


def execute(command, *, output, cwd, env, timeout, max_output_bytes=64_000_000):
    """Bound the subprocess group, including when interrupted or timed out."""
    with output.open('wb') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=cwd, env={**os.environ, **env}, start_new_session=True)
        start = time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic() - start > timeout:
                    raise TimeoutError('capture timeout')
                if output.stat().st_size > max_output_bytes:
                    raise ValueError('capture output limit reached')
                time.sleep(.1)
            if process.returncode:
                raise RuntimeError('capture process failed; inspect the local raw report')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL);process.wait()


def process_request(path, directory, adapter, *, ncu='ncu', timeout=120, minimum_free_bytes=1_073_741_824, runner=execute):
    directory = Path(directory)
    request = validate_request(path, directory)
    if request is None:
        return 'waiting_for_checkpoint'
    receipt_path = directory / 'receipts' / (request['key'] + '.json')
    if receipt_path.exists():
        return 'already_attempted'
    if not adapter.ready(request):
        return 'not_ready'
    if shutil.disk_usage(directory).free < minimum_free_bytes:
        return 'insufficient_storage'
    executable = shutil.which(ncu)
    if not executable:
        return 'ncu_unavailable'
    version = subprocess.run([executable, '--version'], capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip().splitlines()[-1]
    receipt = {'key':request['key'], 'run_id':request['run_id'], 'update':request['update'],
               'config_sha256':request['config_sha256'], 'started_at':time.time(), 'status':'RUNNING'}
    atomic(receipt_path, receipt)
    raw = directory / 'raw_profiles' / (request['key'] + '.csv')
    raw.parent.mkdir(parents=True, exist_ok=True)
    try:
        # __exit__ owns resume/release on success, exceptions and SIGTERM.
        with adapter.capture_context(request) as plan:
            if not isinstance(plan, CapturePlan):
                raise TypeError('adapter must yield CapturePlan')
            if (not plan.command or not all(isinstance(x,str) and x for x in plan.command) or
                not re.fullmatch(r'GPU-[a-fA-F0-9-]+',plan.gpu_uuid) or not plan.kernel_regex or
                not plan.scope or type(plan.launch_count) is not int or not 1<=plan.launch_count<=64 or
                type(plan.launch_skip) is not int or plan.launch_skip<0):
                raise ValueError('incomplete replay plan')
            command = [executable, '--target-processes', 'all', '--csv', '--page', 'raw',
                '--kernel-name-base', 'demangled', '--kernel-name', 'regex:' + plan.kernel_regex,
                '--launch-count', str(plan.launch_count), '--launch-skip', str(plan.launch_skip),
                '--metrics', ','.join([m[2] for m in METRICS.values()] + ['gpu__time_duration.sum']),
                *plan.command]
            runner(command, output=raw, cwd=plan.cwd,
                   env={**plan.env, 'CUDA_VISIBLE_DEVICES':plan.gpu_uuid, 'LC_ALL':'C'}, timeout=timeout)
            # Import in the protected context: even an invalid CSV must release/resume.
            from datetime import datetime, timezone
            capture = import_rows(raw.read_text(), {'run_id':request['run_id'],
                'config_sha256':request['config_sha256'], 'captured_at':datetime.now(timezone.utc).isoformat(),
                'gpu_uuid':plan.gpu_uuid, 'tool_version':version,
                'checkpoint_update':request['update'], 'scope':plan.scope})
            atomic(directory / 'nsight' / (request['key'] + '.json'), capture)
        receipt.update(status='COMPLETE',finished_at=time.time())
    except BaseException as exc:
        receipt.update(status='FAILED',finished_at=time.time(),error_type=type(exc).__name__)
        atomic(receipt_path,receipt)
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
        return 'failed'
    atomic(receipt_path, receipt)
    return 'complete'


def load_adapter(path):
    spec=importlib.util.spec_from_file_location('ml_monitor_user_adapter',path)
    if spec is None or spec.loader is None:raise ValueError('adapter must be a Python file')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not callable(getattr(module,'ready',None)) or not callable(getattr(module,'capture_context',None)):
        raise ValueError('adapter must define ready(request) and capture_context(request)')
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',required=True,type=Path)
    parser.add_argument('--adapter',required=True,type=Path)
    parser.add_argument('--ncu',default='ncu')
    parser.add_argument('--timeout',type=int,default=120)
    parser.add_argument('--minimum-free-gib',type=float,default=1)
    parser.add_argument('--recover',action='store_true',help='invoke adapter recovery for interrupted RUNNING receipts; does not retry captures')
    args=parser.parse_args()
    if args.timeout<1 or args.minimum_free_gib<0:parser.error('invalid timeout/storage bound')
    directory=args.run_directory.resolve()
    if not directory.is_dir():parser.error('run directory does not exist')
    adapter=load_adapter(args.adapter.resolve())
    def interrupted(*_):raise KeyboardInterrupt('capture worker interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    with (directory/'.profile-worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another worker owns this run')
        for path in sorted((directory/'requests').glob('*.json')):
            if args.recover:
                receipt_path=directory/'receipts'/path.name;receipt=read(receipt_path)
                if receipt.get('status')=='RUNNING':
                    if not callable(getattr(adapter,'recover',None)):
                        raise SystemExit('Interrupted capture requires adapter.recover(request, receipt)')
                    adapter.recover(read(path),receipt)
                    atomic(receipt_path,{**receipt,'status':'RECOVERED','finished_at':time.time()})
            else:
                try:
                    result=process_request(path,directory,adapter,ncu=args.ncu,timeout=args.timeout,
                        minimum_free_bytes=int(args.minimum_free_gib*1024**3))
                except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
                    result='request_error:'+type(exc).__name__
                print(path.stem,result,flush=True)


if __name__=='__main__':main()
