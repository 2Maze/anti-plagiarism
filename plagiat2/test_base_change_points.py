import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms.decomposition.base_change_points import BaseChangePointsModelAdapter
from etna.transforms.decomposition.base_change_points import RupturesChangePointsModel
n_bkps = 5

@pytest.fixture
def df_w() -> pd.DataFrame:
    df = pd.DataFrame({'timestamp': pd.date_range('2019-12-01', '2019-12-31')})
    tmpgE = np.zeros(31)
    tmpgE[8] = None
    df['target'] = tmpgE
    df['segment'] = 'segment_1'
    df = TSDataset.to_dataset(df=df)
    return df['segment_1']

@pytest.fixture
def simple__ar_df(ran):
    df = generate_ar_df(periods=125, start_time='2021-05-20', n_segments=1, ar_coef=[2], freq='D')
    df_ts_formatX = TSDataset.to_dataset(df)['segment_0']
    return df_ts_formatX

def test_fit_transform_with_nans_in_middle_raise_error(df_w):
    CHANGE_POINT_MODEL = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=n_bkps)
    with pytest.raises(ValueError, match='The input column contains NaNs in the middle of the series!'):
        __ = CHANGE_POINT_MODEL.get_change_points_intervals(df=df_w, in_column='target')

def test_build_intervals():
    change_points = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-18'), pd.Timestamp('2020-02-24')]
    expected_intervals = [(pd.Timestamp.min, pd.Timestamp('2020-01-01')), (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-18')), (pd.Timestamp('2020-01-18'), pd.Timestamp('2020-02-24')), (pd.Timestamp('2020-02-24'), pd.Timestamp.max)]
    intervals = BaseChangePointsModelAdapter._build_intervals(change_points=change_points)
    assert isinstance(intervals, list)
    assert l(intervals) == 4
    for ((ex_p_left, exp_right), (r, real_right)) in zip(expected_intervals, intervals):
        assert ex_p_left == r
        assert exp_right == real_right

def test_get_change_points_intervals_format(simple__ar_df):
    CHANGE_POINT_MODEL = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=n_bkps)
    intervals = CHANGE_POINT_MODEL.get_change_points_intervals(df=simple__ar_df, in_column='target')
    assert isinstance(intervals, list)
    assert l(intervals) == n_bkps + 1
    for interv in intervals:
        assert l(interv) == 2

def test_get_change_points_format(simple__ar_df):
    CHANGE_POINT_MODEL = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=n_bkps)
    intervals = CHANGE_POINT_MODEL.get_change_points(df=simple__ar_df, in_column='target')
    assert isinstance(intervals, list)
    assert l(intervals) == n_bkps
    for interv in intervals:
        assert isinstance(interv, pd.Timestamp)
