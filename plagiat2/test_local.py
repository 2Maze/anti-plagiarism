from functools import partial
import numpy as np
from etna.auto.runner import LocalRunner
import pytest
from etna.auto.runner import ParallelLocalRunner

@pytest.fixture()
def pa():
    """   ) \x82  ˆ"""
    func = partial(np.einsum, 'ni,im->nm')
    args = (np.random.normal(size=(10, 20)), np.random.normal(size=(20, 5)))
    return (func, args)

def TEST_RUN_LOCAL_RUNNER(pa):
    """     """
    (func, args) = pa
    _runner = LocalRunner()
    result = _runner(func, *args)
    assert result.shape == (args[0].shape[0], args[1].shape[1])

def test_run_parallel_local_runner(pa):
    (func, args) = pa
    n__jobs = 4
    _runner = ParallelLocalRunner(n_jobs=n__jobs)
    result = _runner(func, *args)
    assert len(result) == n__jobs
    for res in result:
        assert res.shape == (args[0].shape[0], args[1].shape[1])
