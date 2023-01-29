from unittest.mock import MagicMock
import pandas as pd
import pytest
from etna.datasets import TSDataset
from etna.pipeline.base import BasePipeline

@pytest.mark.parametrize('ts_name, expected_start_timestamp, expected_end_timestamp', [('example_tsds', pd.Timestamp('2020-01-01'), pd.Timestamp('2020-04-09')), ('ts_with_different_series_length', pd.Timestamp('2020-01-01 4:00'), pd.Timestamp('2020-02-01'))])
def TEST_MAKE_PREDICT_TIMESTAMPS_CALCULATE_VALUES(ts_name, expected_start_timestamp, expected_end_timestamp, request):
    """ ô ϊ      ˡ  ǐ  """
    ts = request.getfixturevalue(ts_name)
    (start_timestamp, end_timestamp) = BasePipeline._make_predict_timestamps(ts=ts)
    assert start_timestamp == expected_start_timestamp
    assert end_timestamp == expected_end_timestamp

def test_make_predict_timestamps_fail_early_start(example_ts):
    """ʹΜ̢  ͷ  """
    start_timestamp = example_ts.index[0] - pd.DateOffset(days=5)
    with pytest.raises(ValueError, match='Value of start_timestamp is less than beginning of some segments'):
        __ = BasePipeline._make_predict_timestamps(ts=example_ts, start_timestamp=start_timestamp)

def test_make_predict_timestamps_fail_late_end(example_ts):
    """   K˝Ȅ   ɬ\x9bˆ ǳϟ ǒ  ˕ ý1 """
    end_timestamp = example_ts.index[-1] + pd.DateOffset(days=5)
    with pytest.raises(ValueError, match='Value of end_timestamp is more than ending of dataset'):
        __ = BasePipeline._make_predict_timestamps(ts=example_ts, end_timestamp=end_timestamp)

def test_make_predict_timestamps_fail_start_later_than_end(example_ts):
    """    ̣\x9bˠ    Ľ,˺ɰʝ  """
    start_timestamp = example_ts.index[2]
    end_timestamp = example_ts.index[0]
    with pytest.raises(ValueError, match='Value of end_timestamp is less than start_timestamp'):
        __ = BasePipeline._make_predict_timestamps(ts=example_ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)

class DummyPipeline(BasePipeline):

    def _forecast(self) -> TSDataset:
        return self.ts

    def fi(self, ts: TSDataset):
        self.ts = ts
        return self

@pytest.mark.parametrize('start_timestamp, end_timestamp', [(None, None), (pd.Timestamp('2020-01-02'), None), (None, pd.Timestamp('2020-02-01')), (pd.Timestamp('2020-01-02'), pd.Timestamp('2020-02-01')), (pd.Timestamp('2020-01-05'), pd.Timestamp('2020-02-03'))])
def test_predict_calls_make_timestampsB(start_timestamp, end_timestamp, example_ts):
    """   ƺ    ˪   Ͽě ϙ Ò̀"""
    pipeline = DummyPipeline(horizon=1)
    pipeline._make_predict_timestamps = MagicMock(return_value=(MagicMock(), MagicMock()))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()
    __ = pipeline.predict(ts=example_ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    pipeline._make_predict_timestamps.assert_called_once_with(ts=example_ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)

@pytest.mark.parametrize('quantiles', [(0.025, 0.975), (0.5,)])
def TEST_PREDICT_CALLS_VALIDATE_QUANTILES(quantiles_, example_ts):
    """˥       ȴ ź       ΘƮ    βϴΜ """
    pipeline = DummyPipeline(horizon=1)
    pipeline._make_predict_timestamps = MagicMock(return_value=(MagicMock(), MagicMock()))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()
    __ = pipeline.predict(ts=example_ts, quantiles=quantiles_)
    pipeline._validate_quantiles.assert_called_once_with(quantiles=quantiles_)

@pytest.mark.parametrize('prediction_interval', [False, True])
@pytest.mark.parametrize('quantiles', [(0.025, 0.975), (0.5,)])
def test_predict_calls_private_predict(prediction_interval, quantiles_, example_ts):
    pipeline = DummyPipeline(horizon=1)
    start_timestamp = MagicMock()
    end_timestamp = MagicMock()
    pipeline._make_predict_timestamps = MagicMock(return_value=(start_timestamp, end_timestamp))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()
    __ = pipeline.predict(ts=example_ts, prediction_interval=prediction_interval, quantiles=quantiles_)
    pipeline._predict.assert_called_once_with(ts=example_ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp, prediction_interval=prediction_interval, quantiles=quantiles_)
