import unittest
from scripts.summarize_trace import summarize, union


def event(name, cat, ts, dur, pid=1, tid=1):
    return dict(name=name, cat=cat, ts=ts, dur=dur, ph='X', pid=pid, tid=tid)


class TraceTests(unittest.TestCase):
    def test_overlap_clipping_and_annotations(self):
        trace = {'traceEvents': [event('perf::step','user_annotation',0,100),
            event('scope','gpu_user_annotation',0,100),
            event('a','kernel',10,30), event('b','kernel',30,30),
            event('copy','gpu_memcpy',90,30)]}
        result = summarize(trace)['steps'][0]
        self.assertEqual(result['device_active_fraction'], {'1': .6})
        self.assertEqual(len(result['top_device_events']), 3)

    def test_devices_not_summed_and_missing_cuda_explicit(self):
        trace={'traceEvents':[event('perf::step','user_annotation',0,100),
            event('a','kernel',0,100,pid=2),event('b','kernel',0,100,pid=3)]}
        self.assertEqual(summarize(trace)['steps'][0]['device_active_fraction'], {'2':1,'3':1})
        self.assertFalse(summarize({'traceEvents':trace['traceEvents'][:1]})['device_events_present'])
        with self.assertRaises(ValueError):summarize({'traceEvents':[]})

    def test_nested_intervals(self):
        self.assertEqual(union([(1,5),(2,3),(4,6),(8,10)]),7)

    def test_events_are_clipped_to_step_boundaries(self):
        trace = {'traceEvents': [event('perf::step', 'user_annotation', 10, 20),
            event('early', 'kernel', 0, 15), event('late', 'gpu_memcpy', 25, 15)]}
        result = summarize(trace)['steps'][0]
        self.assertEqual(result['device_active_fraction'], {'1': .5})

    def test_device_argument_takes_precedence_over_process_id(self):
        kernel = event('kernel', 'kernel', 0, 100, pid=99)
        kernel['args'] = {'device': 3}
        trace = {'traceEvents': [event('perf::step', 'user_annotation', 0, 100), kernel]}
        self.assertEqual(summarize(trace)['steps'][0]['device_active_fraction'], {'3': 1})

    def test_non_duration_events_are_ignored(self):
        marker = event('marker', 'kernel', 0, 100)
        marker['ph'] = 'i'
        trace = {'traceEvents': [event('perf::step', 'user_annotation', 0, 100), marker]}
        self.assertFalse(summarize(trace)['device_events_present'])


if __name__=='__main__':unittest.main()
