#!/usr/bin/env python3
"""Summarize actual CUDA activity in annotated PyTorch Chrome traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def union(intervals):
    end = None
    total = 0.0
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if end is None or a > end:
            total += b - a
        elif b > end:
            total += b - end
        end = max(end, b) if end is not None else b
    return total


def summarize(trace):
    events = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and e.get('dur', 0) > 0]
    steps = [e for e in events if e.get('name') == 'perf::step']
    if not steps:
        raise ValueError('No perf::step ranges; cannot establish step wall time')
    devices = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    result = []
    for step in steps:
        start, stop = step['ts'], step['ts'] + step['dur']
        intervals = defaultdict(list)
        by_name = defaultdict(lambda: [0, 0.0])
        host = defaultdict(lambda: [0, 0.0])
        for e in devices:
            a, b = max(start, e['ts']), min(stop, e['ts'] + e['dur'])
            if b > a:
                intervals[str(e.get('args', {}).get('device', e.get('pid')))].append((a, b))
                by_name[e['name']][0] += 1
                by_name[e['name']][1] += b-a
        for e in events:
            if e.get('tid') != step.get('tid') or e.get('pid') != step.get('pid'):
                continue
            if e['ts'] < start or e['ts'] + e['dur'] > stop + 1:
                continue
            if e['name'].startswith('perf::') and e['name'] != 'perf::step' or e.get('cat') == 'cuda_runtime':
                host[e['name']][0] += 1
                host[e['name']][1] += e['dur']
        result.append({'wall_ms': step['dur']/1000,
                       'device_active_ms': {k:union(v)/1000 for k,v in intervals.items()},
                       'device_active_fraction': {k:union(v)/step['dur'] for k,v in intervals.items()},
                       'top_device_events': [{'name':k,'count':v[0],'summed_ms':v[1]/1000} for k,v in sorted(by_name.items(),key=lambda x:x[1][1],reverse=True)[:15]],
                       'inclusive_host_ranges': {k:{'count':v[0],'ms':v[1]/1000} for k,v in host.items()}})
    return {'steps': result, 'device_events_present': bool(devices),
            'interpretation': 'Device interval unions per device, not occupancy. Host ranges are inclusive and may overlap GPU activity; do not add them to device time.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(summarize(json.loads(a.trace.read_text())),indent=2)+'\n')
