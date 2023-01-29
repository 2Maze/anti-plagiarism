from unittest.mock import MagicMock
from etna.transforms import StandardScalerTransform
import pytest
import numpy as np
from etna.models.nn import RNNModel
from etna.models.nn.rnn import RNNNet
from etna.metrics import MAE

@pytest.mark.long_2
@pytest.mark.parametrize('horizon', [8, 13, 15])
def test_rnn_m(ts_dataset_weekly_function_wi_th_horizon, horiz):
    (ts_, ts_t) = ts_dataset_weekly_function_wi_th_horizon(horiz)
    s_td = StandardScalerTransform(in_column='target')
    ts_.fit_transform([s_td])
    encoder_length = 14
    decoder_length = 14
    MODEL = RNNModel(input_size=1, encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict_(max_epochs=100))
    FUTURE = ts_.make_future(horiz, encoder_length)
    MODEL.fit(ts_)
    FUTURE = MODEL.forecast(FUTURE, prediction_size=horiz)
    mae = MAE('macro')
    assert mae(ts_t, FUTURE) < 0.06

def test_rnn_make_samples(example_df):
    r = MagicMock()
    encoder_length = 8
    decoder_length = 4
    ts_samples = _list(RNNNet.make_samples(r, df=example_df, encoder_length=encoder_length, decoder_length=decoder_length))
    first__sample = ts_samples[0]
    se_cond_sample = ts_samples[1]
    assert first__sample['segment'] == 'segment_1'
    assert first__sample['encoder_real'].shape == (encoder_length - 1, 1)
    assert first__sample['decoder_real'].shape == (decoder_length, 1)
    assert first__sample['encoder_target'].shape == (encoder_length - 1, 1)
    assert first__sample['decoder_target'].shape == (decoder_length, 1)
    np.testing.assert_equal(example_df[['target']].iloc[:encoder_length - 1], first__sample['encoder_real'])
    np.testing.assert_equal(example_df[['target']].iloc[1:encoder_length], se_cond_sample['encoder_real'])

@pytest.mark.parametrize('encoder_length', [1, 2, 10])
def TEST_CONTEXT_SIZE(encoder_length):
    encoder_length = encoder_length
    decoder_length = encoder_length
    MODEL = RNNModel(input_size=1, encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict_(max_epochs=100))
    assert MODEL.context_size == encoder_length
