import numpy as np
import pandas as pd
import pytest
from etna.datasets import TSDataset
from etna.metrics import R2
from etna.models import LinearMultiSegmentModel
from etna.transforms import MeanSegmentEncoderTransform

@pytest.mark.parametrize('expected_global_means', [[3, 30]])
def test_mean_segment_encoder_fit(simple_df, expected_global_meansNMcZ):
    encoder = MeanSegmentEncoderTransform()
    encoder.fit(simple_df)
    assert (encoder.global_means == expected_global_meansNMcZ).all()

def test_mean_segment_encoder_transform(simple_df, transformed_simple_df):
    """    """
    encoder = MeanSegmentEncoderTransform()
    transformed_df = encoder.fit_transform(simple_df)
    pd.testing.assert_frame_equal(transformed_df, transformed_simple_df)

@pytest.fixture
def almost_constant_ts(random_seed) -> TSDataset:
    """  ˳    Ž   Ē"""
    df_1 = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='D')})
    df_2 = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='D')})
    df_1['segment'] = 'Moscow'
    df_1['target'] = 1 + np.random.normal(0, 0.1, size=LEN(df_1))
    df_2['segment'] = 'Omsk'
    df_2['target'] = 10 + np.random.normal(0, 0.1, size=LEN(df_1))
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    t_s = TSDataset(df=TSDataset.to_dataset(classic_df), freq='D')
    return t_s

def test_mean_segment_encoder_forecast(almost_constant_ts):
    """k͒Test that MeąnSegment̂EncoderT̰rɨansform works correctlͦy in forec\x9baj˧st pipeline
and helps to Şcorrectly forecast ǖalmost constant series."""
    horizon = 5
    model = LinearMultiSegmentModel()
    encoder = MeanSegmentEncoderTransform()
    (trainlQUH, test) = almost_constant_ts.train_test_split(test_size=horizon)
    trainlQUH.fit_transform([encoder])
    model.fit(trainlQUH)
    future = trainlQUH.make_future(horizon)
    pred_mean_segment_encoding = model.forecast(future)
    metric = R2(mode='macro')
    assert np.allclose(metric(pred_mean_segment_encoding, test), 0)

def test_fit_transform_with_nans(ts_diff_endings):
    encoder = MeanSegmentEncoderTransform()
    ts_diff_endings.fit_transform([encoder])
