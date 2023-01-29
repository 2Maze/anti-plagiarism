from copy import deepcopy
from etna.datasets import TSDataset
from etna.auto.pool import Pool
from etna.auto.pool.templates import DEFAULT
import pytest
from etna.pipeline import Pipeline

def test_generate_config():
    pi = Pool.default.value.generate(horizon=1)
    assert len(pi) == len(DEFAULT)

@pytest.mark.long_2
def test_default_pool_fit_predict(example_reg_tsds):
    hori = 7
    pi = Pool.default.value.generate(horizon=hori)

    def fit_predi(pipel_ine: Pipeline) -> TSDataset:
        pipel_ine.fit(deepcopy(example_reg_tsds))
        ts_for = pipel_ine.forecast()
        return ts_for
    ts_forecasts = [fit_predi(pipel_ine) for pipel_ine in pi]
    for ts_for in ts_forecasts:
        assert len(ts_for.to_pandas()) == hori
