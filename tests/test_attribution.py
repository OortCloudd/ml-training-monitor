import csv
import io
import unittest
from mlmonitor.step_attribution import summarize
from mlmonitor.import_ncu import import_rows


class Event:
    def __init__(self,name,kind,start,end,annotation=False):
        self.n,self.k,self.a,self.b,self.note=name,kind,start*1_000_000,end*1_000_000,annotation
    def name(self):return self.n
    def device_type(self):return self.k
    def start_ns(self):return self.a
    def end_ns(self):return self.b
    def is_user_annotation(self):return self.note


class TimelineTests(unittest.TestCase):
    def test_cuda_launch_and_sync_are_measured_without_gpu_double_count(self):
        s=summarize(0,100_000_000,[('forward_host',0,100_000_000)],
                    [Event('cudaLaunchKernel','CPU',0,30),Event('gemm','CUDA',20,80),
                     Event('cudaStreamSynchronize','CPU',70,90)])
        self.assertEqual(s['exclusive_ms']['launch_api'],20)
        self.assertEqual(s['exclusive_ms']['gpu_kernels'],60)
        self.assertEqual(s['exclusive_ms']['sync_api'],10)
        self.assertEqual(s['exclusive_ms']['forward_host'],10)
        self.assertEqual(sum(s['family_ms'].values()),100)
        self.assertEqual(s['runtime_events'],2)

    def test_copy_directions_and_annotations(self):
        s=summarize(0,100_000_000,[],[Event('Memcpy HtoD','CUDA',0,10),Event('Memcpy DtoH','CUDA',10,20),
                    Event('Memcpy DtoD','CUDA',20,30),Event('Memset','CUDA',30,40),
                    Event('probe::range','CUDA',0,100,True)])
        for k in ['copy_h2d','copy_d2h','copy_d2d','memory_set']:
            self.assertEqual(s['exclusive_ms'][k],10)
        self.assertEqual(s['device_events'],4)
        self.assertEqual(s['exclusive_ms']['host_other'],60)

    def test_real_ncu_wide_layout_and_per_cycle_unit(self):
        f=io.StringIO();w=csv.writer(f)
        w.writerow(['ID','Kernel Name','gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed',
                    'smsp__warps_eligible.avg.per_cycle_active'])
        w.writerow(['','','nsecond','%','warp'])
        w.writerow(['0','gemm','2000','83.4','0.27'])
        meta={'schema':'gpu-nsight-capture-v1','run_id':'example_run','captured_at':'2026-09-26T00:00:00+00:00',
              'gpu_uuid':'GPU-abcdef','config_sha256':'a'*64,'tool_version':'fixture','scope':'unit test only'}
        k=import_rows(f.getvalue(),meta)['kernels'][0]
        self.assertEqual(k['duration_us'],2)
        self.assertEqual(k['metrics']['sm_throughput_pct'],83.4)
        self.assertEqual(k['metrics']['eligible_warps'],.27)


if __name__=='__main__':unittest.main()
