import csv,os,shutil,subprocess
from pathlib import Path
from .storage import finite as number

class Hardware:
    def __init__(self, settings):
        self.settings=settings
        self.previous = None

    def read(self):
        values = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        total, idle = sum(values), values[3] + values[4]
        cpu = None
        if self.previous and total > self.previous[0]:
            cpu = 100 * (1 - (idle - self.previous[1]) / (total - self.previous[0]))
        self.previous = total, idle
        mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        gpus, errors = [], []
        try:
            if not self.settings['gpu_enabled']:raise OSError('disabled')
            output = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,utilization.memory,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,uuid,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True).stdout
            for row in csv.reader(output.splitlines()):
                def value(i):
                    try:
                        return number(float(row[i]))
                    except ValueError:
                        return None
                gpus.append(dict(index=int(row[0]), name=row[1].strip(), utilization=value(2), memory_used=value(3),
                                 memory_total=value(4), temperature=value(5), power=value(6), power_limit=value(7),
                                 memory_activity=value(8), sm_clock=value(9), memory_clock=value(10),
                                 sm_clock_max=value(11), memory_clock_max=value(12), uuid=row[13].strip(),
                                 power_cap={'Active':True,'Not Active':False}.get(row[14].strip()),
                                 thermal_cap=any(row[i].strip()=='Active' for i in [15,16])
                                     if all(row[i].strip() in ['Active','Not Active'] for i in [15,16]) else None))
        except (OSError, subprocess.SubprocessError, ValueError):
            errors.append('Mesures GPU indisponibles')
        # Board activity is attributable only when the training PID is the sole
        # compute process on that GPU. Missing/ambiguous ownership stays unknown.
        compute_pids = None
        try:
            if not self.settings['gpu_enabled']:raise OSError('disabled')
            output = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], capture_output=True,
                                    text=True, timeout=3, check=True).stdout
            compute_pids = {}
            for row in csv.reader(output.splitlines()):
                compute_pids.setdefault(row[0].strip(), []).append(int(row[1]))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        for gpu in gpus:
            gpu['compute_pids'] = compute_pids.get(gpu['uuid'], []) if compute_pids is not None else None
        disks = []
        for disk in self.settings['disks']:
            path=disk['path']
            try:
                d = shutil.disk_usage(path)
                disks.append({'path': disk['label'], 'total': d.total, 'free': d.free, 'used': d.used})
            except OSError:
                continue
        if not self.settings['gpu_enabled']:errors=[]
        indices=self.settings['gpu_indices']
        if indices is not None:gpus=[g for g in gpus if g['index'] in indices]
        return {'cpu': cpu, 'cores': os.cpu_count(), 'ram_total': mem['MemTotal'],
                'ram_used': mem['MemTotal'] - mem['MemAvailable'], 'gpus': gpus, 'disks': disks,
                'errors': errors, 'load': list(os.getloadavg())}
