"""Optional CPU qualification in an environment that already has PyTorch."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from mlmonitor import Monitor


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is optional')
class TorchIntegrationTests(unittest.TestCase):
    def test_light_hooks_preserve_model_optimizer_rng_and_loss_exactly(self):
        import torch
        def run(directory=None):
            torch.manual_seed(817)
            model=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.GELU(),torch.nn.Linear(16,3))
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            inputs=torch.randn(4,8);target=torch.randn(4,3)
            monitor=Monitor(directory,run_id='cpu_parity',config={'test':'CPU parity'}) if directory else None
            losses=[]
            for step in range(1,6):
                if monitor:
                    with monitor.step(step) as record:
                        with monitor.phase('forward_host'):output=model(inputs);loss=(output-target).square().mean()
                        with monitor.phase('backward_host'):loss.backward()
                        with monitor.phase('optimizer'):optimizer.step();optimizer.zero_grad(set_to_none=True)
                        record.record(loss=loss.item())
                else:
                    output=model(inputs);loss=(output-target).square().mean();loss.backward()
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
            if monitor:monitor.close()
            return model.state_dict(),optimizer.state_dict(),torch.get_rng_state(),losses
        def equal(a,b):
            if isinstance(a,torch.Tensor):return torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(equal(run(),run(Path(tmp))))


if __name__=='__main__':unittest.main()
