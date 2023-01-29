from typing import Tuple
import numpy as np
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import MAPE
from etna.datasets import TSDataset
from etna.metrics.utils import compute_metrics

def test_compute_metrics(train__test_dfs: Tuple[TSDataset, TSDataset]):
    (forecast_df, t) = train__test_dfs
    metrics = [MAE('per-segment'), MAE(mode='macro'), MSE('per-segment'), MAPE(mode='macro', eps=1e-05)]
    expected_keys = ["MAE(mode = 'per-segment', )", "MAE(mode = 'macro', )", "MSE(mode = 'per-segment', )", "MAPE(mode = 'macro', eps = 1e-05, )"]
    result = compute_metrics(metrics=metrics, y_true=t, y_pred=forecast_df)
    np.testing.assert_array_equal(sorted(expected_keys), sorted(result.keys()))
