"""Gate E candidates against the original path and an explicit tied matrix."""
import importlib.util
from pathlib import Path
import pytest
import torch
from vocab_lm.model import HashVocabulary

spec = importlib.util.spec_from_file_location('gate_e', Path('results/experiment_001/gate_e/benchmark.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

@pytest.mark.parametrize('k', [2, 4])
@pytest.mark.parametrize('project', [bench.vector, bench.chunk, bench.selected])
def test_tied_matrix_and_gradients(k, project):
    torch.manual_seed(42)
    v = HashVocabulary(31, 8, 7, (17,29,43,59)[:k]).double()
    ids = torch.tensor([[0,3,3,30]])
    h = torch.randn(1,4,8,dtype=torch.double,requires_grad=True)
    matrix = (v.components[v.indices]*v.importance.unsqueeze(-1)).sum(1)
    torch.testing.assert_close(v.lookup(ids), matrix[ids])
    torch.testing.assert_close(bench.lookup(v,ids),matrix[ids])
    torch.testing.assert_close(project(v,h),h @ matrix.T)
    actual = project(v,h).square().sum()+bench.lookup(v,ids).square().sum()
    expected = (h @ matrix.T).square().sum()+matrix[ids].square().sum()
    for a,b in zip(torch.autograd.grad(actual,[h,*v.parameters()],retain_graph=True),
                   torch.autograd.grad(expected,[h,*v.parameters()])):
        torch.testing.assert_close(a,b)
