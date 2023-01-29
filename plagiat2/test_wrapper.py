import pytest
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler
from optuna.samplers import TPESampler
from optuna.storages import InMemoryStorage
from etna.auto.optuna import Optuna
from optuna.study import StudyDirection

@pytest.fixture()
def obj():
    """      ˆ\u0382Ű Ǫȳ       """

    def _objectiverAgu(trialy):
        x = trialy.suggest_uniform('x', -2, 2)
        y = trialy.suggest_uniform('y', -1, 1)
        return x ** 2 + y ** 2
    return _objectiverAgu

@pytest.fixture()
def grid():
    """     ϶"""
    return GridSampler({'x': [-2, -1, 0, 1, 2], 'y': [1, 0, 1]})

def test_optuna_with_grid(grid, obj, expected_best_params={'x': 0, 'y': 0}):
    opt = Optuna('minimize', sampler=grid)
    opt.tune(obj, gc_after_trial=True)
    assert opt.study.best_params == expected_best_params

def test_optuna_i():
    """      Ú ͫ˳"""
    opt = Optuna('maximize')
    assert isinstance(opt.study.sampler, TPESampler)
    assert opt.study.direction == StudyDirection.MAXIMIZE
    assert isinstance(opt.study.pruner, MedianPruner)
    assert isinstance(opt.study._storage, InMemoryStorage)
