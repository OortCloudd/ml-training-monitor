from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mlmonitor.advice import ProposalStore, opportunities
from mlmonitor.config import load_config
from mlmonitor.server import Sampler, handler
from mlmonitor.step_attribution import CATEGORIES


def snapshot():
    return {'timestamp':time.time(),'runs':[{'id':'test','name':'Test run','config_sha256':'a'*64,
            'step':20,'seconds':1,'window':100,'telemetry_window':{'records':100,'first_update':1,'last_update':20,
                'valid_samples':{'seconds':100}},'bottleneck':{'status':'disabled'}}],
            'hardware':{'gpus':[{'index':0,'utilization':99,'memory_activity':99}]}}


def plan():
    return {'title':'Compare one input delivery change','target_revision':'fixture-revision',
            'change':'Change only the fixture prefetch depth from one to two prepared batches.',
            'rationale':'Investigate recorded input waiting; this is a synthetic test fixture.',
            'expected_effect':'Less waiting; no measured speedup is claimed.',
            'validation':'Compare the same inputs and completed work; inspect outputs and state.',
            'risks':'Additional host memory for one batch.','rollback':'Restore the original prefetch depth.'}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.store=ProposalStore(self.root/'proposals')

    def test_submit_is_pending_and_keeps_reviewed_evidence(self):
        data=snapshot();record=self.store.submit(data,'test',plan())
        self.assertEqual(record['status'],'pending')
        data['runs'][0]['step']=99
        saved=self.store.get(record['id'])
        self.assertEqual(saved['content']['context']['update'],20)
        self.assertEqual(saved['content']['plan']['target_revision'],'fixture-revision')
        self.assertEqual(saved['reviews'],[])

    def test_exact_proposal_can_be_approved_and_revoked(self):
        data=snapshot();record=self.store.submit(data,'test',plan())
        approved=self.store.review(record['id'],record['content_sha256'],'approved','Try this bounded scope.',data)
        self.assertEqual(approved['status'],'approved')
        self.assertEqual(approved['reviews'][-1]['content_sha256'],record['content_sha256'])
        revoked=self.store.review(record['id'],record['content_sha256'],'revoked','Do not start.',data)
        self.assertEqual(revoked['status'],'revoked')
        with self.assertRaises(ValueError):
            self.store.record_outcome(record['id'],record['content_sha256'],'completed','Not authorized.')

    def test_rejection_cannot_become_an_agent_reported_success(self):
        data=snapshot();record=self.store.submit(data,'test',plan())
        self.store.review(record['id'],record['content_sha256'],'rejected','Keep the current implementation.',data)
        with self.assertRaises(ValueError):
            self.store.record_outcome(record['id'],record['content_sha256'],'completed','Rejected work.')

    def test_changed_content_or_configuration_cannot_be_approved(self):
        data=snapshot();record=self.store.submit(data,'test',plan())
        with self.assertRaises(ValueError):
            self.store.review(record['id'],'b'*64,'approved','',data)
        changed=deepcopy(data);changed['runs'][0]['config_sha256']='c'*64
        with self.assertRaises(ValueError):
            self.store.review(record['id'],record['content_sha256'],'approved','',changed)
        self.assertFalse(self.store.listing(changed)['proposals'][0]['context_matches'])
        path=self.store.path(record['id']);raw=json.loads(path.read_text())
        raw['content']['plan']['change']='Changed after the user loaded the page.';path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):self.store.get(record['id'])

    def test_reported_outcome_keeps_the_approval(self):
        data=snapshot();record=self.store.submit(data,'test',plan())
        with self.assertRaises(ValueError):
            self.store.record_outcome(record['id'],record['content_sha256'],'completed','No approval yet.')
        self.store.review(record['id'],record['content_sha256'],'approved','',data)
        finished=self.store.record_outcome(record['id'],record['content_sha256'],'reverted','Agent reports the change was reverted after the comparison.')
        self.assertEqual(finished['outcome']['source'],'agent_report')
        self.assertEqual(finished['reviews'][0]['decision'],'approved')

    def test_editing_a_status_flag_alone_is_not_a_recorded_approval(self):
        record=self.store.submit(snapshot(),'test',plan())
        path=self.store.path(record['id']);raw=json.loads(path.read_text());raw['status']='approved'
        path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):self.store.get(record['id'])

    def test_no_incomplete_plan_or_unknown_run(self):
        incomplete=plan();incomplete.pop('rollback')
        with self.assertRaises(ValueError):self.store.submit(snapshot(),'test',incomplete)
        with self.assertRaises(ValueError):self.store.submit(snapshot(),'other',plan())
        with self.assertRaises(ValueError):self.store.get('../some-file')


class OpportunityTests(unittest.TestCase):
    def test_high_activity_alone_does_not_claim_compute_or_bandwidth_saturation(self):
        result=opportunities(snapshot())
        self.assertEqual([item['id'] for item in result],['observe_step'])
        self.assertIn('bounded representative capture',result[0]['experiment'])

    def test_aligned_input_wait_keeps_scope_and_sample_count(self):
        data=snapshot();data['runs'][0]['light_profile']={'first_update':1,'last_update':20,'steps':100,
            'captured_at':time.time(),'phases':{'input_wait':{'seconds':50,'steps':100,'calls':100}}}
        result=opportunities(data)
        self.assertEqual(result[0]['id'],'input_wait')
        self.assertEqual(result[0]['evidence']['samples'],100)
        self.assertEqual(result[0]['evidence']['mean_wait_seconds'],.5)
        data['runs'][0]['light_profile']['last_update']=10
        self.assertEqual(opportunities(data)[0]['id'],'observe_step')

    def test_profile_costs_are_ranked_with_dates_not_promised_speedups(self):
        data=snapshot();values=dict.fromkeys(CATEGORIES,0.)
        values.update(gpu_kernels=60,input_wait=30,host_other=10)
        data['runs'][0]['bottleneck']={'capture':{'valid':True,'exclusive_ms':values,'step_ms':100,
            'config_sha256':'a'*64,'captured_at':data['timestamp']-60,'update':10}}
        result=opportunities(data)
        self.assertEqual(result[0]['id'],'profile_kernels')
        self.assertEqual(result[0]['evidence']['milliseconds'],60)
        self.assertEqual(result[0]['evidence']['update'],10)
        self.assertIn('not a measured speedup',result[0]['note'])
        data['runs'][0]['bottleneck']['capture']['config_sha256']='b'*64
        self.assertEqual(opportunities(data)[0]['id'],'observe_step')


class ReviewHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def make_sampler(self,enabled):
        path=self.root/'config.json';path.write_text(json.dumps({'gpu_enabled':False,'runs':[],
            'decision_support':{'enabled':enabled,'directory':'proposals'}}))
        sampler=Sampler(load_config(path));sampler.snapshot=snapshot()
        return sampler

    def serve(self,sampler):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(sampler))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(lambda:(server.shutdown(),server.server_close(),thread.join()))
        return f'http://127.0.0.1:{server.server_port}'

    def post(self,base,payload,token=None,origin=None):
        headers={'Content-Type':'application/json'}
        if token is not None:headers['X-Review-Token']=token
        if origin is not None:headers['Origin']=origin
        with urlopen(Request(base+'/api/proposals/review',data=json.dumps(payload).encode(),headers=headers,method='POST')) as response:
            return json.load(response)

    def test_disabled_mode_has_no_store_and_no_review_route(self):
        sampler=self.make_sampler(False);self.assertIsNone(sampler.proposals)
        self.assertFalse((self.root/'proposals').exists())
        base=self.serve(sampler)
        with urlopen(base+'/api/decisions') as response:self.assertEqual(json.load(response),{'enabled':False})
        with self.assertRaises(HTTPError) as error:self.post(base,{})
        self.assertEqual(error.exception.code,403)

    def test_review_requires_page_token_origin_and_exact_content(self):
        sampler=self.make_sampler(True);base=self.serve(sampler)
        record=sampler.proposals.submit(sampler.snapshot,'test',plan())
        payload={'id':record['id'],'content_sha256':record['content_sha256'],'decision':'approved'}
        for token,origin in [(None,base),(sampler.review_token,'https://unrelated.example'),('wrong',base)]:
            with self.subTest(origin=origin),self.assertRaises(HTTPError) as error:self.post(base,payload,token,origin)
            self.assertEqual(error.exception.code,403)
        with urlopen(base+'/api/decisions') as response:
            text=response.read().decode();self.assertNotIn(sampler.review_token,text)
        approved=self.post(base,payload,sampler.review_token,base)
        self.assertEqual(approved['status'],'approved')
        with self.assertRaises(HTTPError) as error:self.post(base,payload,sampler.review_token,base)
        self.assertEqual(error.exception.code,409)


if __name__=='__main__':unittest.main()
