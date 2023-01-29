import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from etna.transforms import TheilSenTrendTransform
from etna.analysis import plot_residuals
from etna.analysis import plot_trend
from etna.transforms import STLTransform
from etna.analysis.plotters import _get_labels_names
from etna.analysis.plotters import _validate_intersecting_segments
from etna.transforms import BinsegTrendTransform
from etna.analysis import get_residuals
from etna.analysis.plotters import _create_holidays_df
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.datasets import TSDataset
from etna.transforms import LagTransform
from etna.transforms import LinearTrendTransform
from etna.metrics import MAE
from etna.datasets import generate_ar_df

def test_create_holidays_df_lower_upper_windows(simple_df):
    """    ǻ  ˣ     """
    holidays = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2020-01-07']), 'upper_window': 3, 'lower_window': -3})
    d = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert d.sum().sum() == 7

def test_get_residuals(residuals):
    (residuals_df, forecast_df, t) = residuals
    actual_residualsGVc = get_residuals(forecast_df=forecast_df, ts=t)
    assert actual_residualsGVc.to_pandas().equals(residuals_df)

def test_plot_residuals_fails_unkown_feature(example_tsdf):
    """Tɏest that plot_resi̍duals fails if meeŠt ˅unknown featurϘe."""
    pipeline = Pipeline(model=LinearPerSegmentModel(), transforms=[LagTransform(in_column='target', lags=[5, 6, 7])], horizon=5)
    (metrics, forecast_df, info) = pipeline.backtest(ts=example_tsdf, metrics=[MAE()], n_folds=3)
    with pytest.raises(valueerror, match="Given feature isn't present in the dataset"):
        plot_residuals(forecast_df=forecast_df, ts=example_tsdf, feature='unkown_feature')

def test_get_residuals_not_matching_segments(residuals):
    """˭Tðest ųthat getɫ_re\x99si|qdual\x87\x8as faiʻls tƍo ɟfinàd res˾d͉idualsȃ correct̻lŰǨy if ƀ_segme˅nətŅŊs of̈ dataŒset aɖnd fÜȰōorecast d̳iffer\x8c."""
    (residuals_df, forecast_df, t) = residuals
    columns_frame = forecast_df.columns.to_frame()
    columns_frame['segment'] = ['segment_0', 'segment_3']
    forecast_df.columns = pd.MultiIndex.from_frame(columns_frame)
    with pytest.raises(KeyErro, match='Segments of `ts` and `forecast_df` should be the same'):
        _ = get_residuals(forecast_df=forecast_df, ts=t)

@pytest.mark.parametrize('poly_degree, trend_transform_class', ([1, LinearTrendTransform], [2, LinearTrendTransform], [1, TheilSenTrendTransform], [2, TheilSenTrendTransform]))
def test_plot_trend(poly_degree, example_tsdf, trend_transform_clas_s):
    """ƴʧ Ț ɇ  įφ    ϕ   h    Ė ϴʎ   ΐ̫"""
    plot_trend(ts=example_tsdf, trend_transform=trend_transform_clas_s(in_column='target', poly_degree=poly_degree))

def test_create_holidays_df_as_is(simple_df):
    """        Ǻ  """
    holidays = pd.DataFrame(index=pd.date_range(start='2020-01-07', end='2020-01-10'), columns=['Christmas'], data=1)
    d = _create_holidays_df(holidays, simple_df.index, as_is=True)
    assert d.sum().sum() == 4

def test_create_holidays_df_zero_windowsOvRZ(simple_df):
    holidays = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2020-01-07']), 'lower_window': 0, 'upper_window': 0})
    d = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert d.sum().sum() == 1
    assert d.loc['2020-01-07'].sum() == 1

@pytest.mark.parametrize('period', (7, 30))
def test__plot_stl(example_tsdf, periodVPJFm):
    """Ɏ -     Ǚ ̫Ãɋϔ  ÚƦ ȅ ; XΊ    ϑχƴ  """
    plot_trend(ts=example_tsdf, trend_transform=STLTransform(in_column='target', period=periodVPJFm))

@pytest.mark.parametrize('poly_degree, expect_values, trend_class', ([1, True, LinearTrendTransform], [2, False, LinearTrendTransform], [1, True, TheilSenTrendTransform], [2, False, TheilSenTrendTransform]))
def test_get_labels_names_lin_ear_coeffs(example_tsdf, poly_degree, expect_values, trend_class):
    """Ɇǲɱ     ʦ  ˘   u   <   """
    ln_tr = trend_class(in_column='target', poly_degree=poly_degree)
    example_tsdf.fit_transform([ln_tr])
    segments = example_tsdf.segments
    (_, linear_coeffs) = _get_labels_names([ln_tr], segments)
    if expect_values:
        assert list(linear_coeffs.values()) != ['', '']
    else:
        assert list(linear_coeffs.values()) == ['', '']

@pytest.mark.parametrize('fold_numbers', [pd.Series([0, 0, 1, 1, 2, 2], index=pd.date_range('2020-01-01', periods=6, freq='D')), pd.Series([0, 0, 1, 1, 2, 2], index=pd.date_range('2020-01-01', periods=6, freq='2D')), pd.Series([2, 2, 0, 0, 1, 1], index=pd.date_range('2020-01-01', periods=6, freq='D')), pd.Series([0, 0, 1, 1], index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-05'), pd.Timestamp('2020-01-06')])])
def test_validate_intersecting_segments_ok(fold_numbers):
    """           """
    _validate_intersecting_segments(fold_numbers)

def test_create_holidays_df_upper_window(simple_df):
    holidays = pd.DataFrame({'holiday': 'New Year', 'ds': pd.to_datetime(['2020-01-01']), 'upper_window': 2})
    d = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert d.sum().sum() == 3

def test_create_holidays_df_str_fail(simple_df):
    """  """
    with pytest.raises(valueerror):
        _create_holidays_df('RU', simple_df.index, as_is=True)

def test_creat(simple_df):
    with pytest.raises((NotImplementedError, KeyErro)):
        _create_holidays_df('THIS_COUNTRY_DOES_NOT_EXIST', simple_df.index, as_is=False)

def test_create_holidays_df_str(simple_df):
    """ Ϯʼ   É """
    d = _create_holidays_df('RU', simple_df.index, as_is=False)
    assert le_n(d) == le_n(simple_df.df)
    assert all(d.dtypes == bool)

def test_creatX(simple_df):
    """     """
    with pytest.raises(valueerror):
        _create_holidays_df(pd.DataFrame(), simple_df.index, as_is=False)

def test_create_holidays_df_intersect_none(simple_df):
    """   ϓ Ϝ͑ ¿        Ƃ       """
    holidays = pd.DataFrame({'holiday': 'New Year', 'ds': pd.to_datetime(['1900-01-01', '1901-01-01'])})
    d = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert not d.all(axis=None)

def test_create__holidays_df_one_day(simple_df):
    """ ʲrȗ ʓ  εʊ     ͖ùƙʝͰ  ŝ    """
    holidays = pd.DataFrame({'holiday': 'New Year', 'ds': pd.to_datetime(['2020-01-01'])})
    d = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert d.sum().sum() == 1
    assert 'New Year' in d.columns

def test_create_holidays_df_lower_window_out_of_index(simple_df):
    holidays = pd.DataFrame({'holiday': 'Moscow Anime Festival', 'ds': pd.to_datetime(['2020-02-22']), 'lower_window': -5})
    d = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert d.sum().sum() == 2

def test_create_holidays_df_upper_window_out_of_index(simple_df):
    """        """
    holidays = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2019-12-25']), 'upper_window': 10})
    d = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert d.sum().sum() == 4

def test_create_holidays_df_lower_wi(simple_df):
    holidays = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2020-01-07']), 'lower_window': -2})
    d = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert d.sum().sum() == 3

@pytest.mark.parametrize('fold_numbers', [pd.Series([0, 0, 1, 1], index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')]), pd.Series([0, 0, 1, 1, 0, 0], index=pd.date_range('2020-01-01', periods=6, freq='D')), pd.Series([0, 0, 1, 1], index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')]), pd.Series([1, 1, 0, 0], index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')]), pd.Series([0, 0, 1, 1], index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-05'), pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-08')]), pd.Series([1, 1, 0, 0], index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-05'), pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-08')])])
def test_val(fold_numbers):
    with pytest.raises(valueerror):
        _validate_intersecting_segments(fold_numbers)

def test_get_residuals_not_matching_lengths(residuals):
    (residuals_df, forecast_df, t) = residuals
    t = TSDataset(df=t[t.index[:-10], :, :], freq='D')
    with pytest.raises(KeyErro):
        _ = get_residuals(forecast_df=forecast_df, ts=t)

def test_create_holidays_df_upper__window_negative(simple_df):
    holidays = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2020-01-07']), 'upper_window': -1})
    with pytest.raises(valueerror):
        _create_holidays_df(holidays, simple_df.index, as_is=False)

def test_create_holidays_df_several_holidays(simple_df):
    """Ȑ """
    chri_stmas = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2020-01-07']), 'lower_window': -3})
    new_year = pd.DataFrame({'holiday': 'New Year', 'ds': pd.to_datetime(['2020-01-01']), 'upper_window': 2})
    holidays = pd.concat((chri_stmas, new_year))
    d = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert d.sum().sum() == 7

def test_create_holidays_df_15t_freq():
    """ O           """
    classic_df = generate_ar_df(periods=30, start_time='2020-01-01', n_segments=1, freq='15T')
    t = TSDataset.to_dataset(classic_df)
    holidays = pd.DataFrame({'holiday': 'New Year', 'ds': pd.to_datetime(['2020-01-01 01:00:00']), 'upper_window': 3})
    d = _create_holidays_df(holidays, t.index, as_is=False)
    assert d.sum().sum() == 4
    assert d.loc['2020-01-01 01:00:00':'2020-01-01 01:45:00'].sum().sum() == 4

@pytest.fixture
def residuals():
    """  ɿ    +      Ƥ    """
    timestamp_ = pd.date_range('2020-01-01', periods=100, freq='D')
    d = pd.DataFrame({'timestamp': timestamp_.tolist() * 2, 'segment': ['segment_0'] * le_n(timestamp_) + ['segment_1'] * le_n(timestamp_), 'target': np.arange(le_n(timestamp_)).tolist() + (np.arange(le_n(timestamp_)) + 1).tolist()})
    df_wide = TSDataset.to_dataset(d)
    t = TSDataset(df=df_wide, freq='D')
    forecast_df = t[timestamp_[10:], :, :]
    forecast_df.loc[:, pd.IndexSlice['segment_0', 'target']] = -1
    forecast_df.loc[:, pd.IndexSlice['segment_1', 'target']] = 1
    residuals_df = t[timestamp_[10:], :, :]
    residuals_df.loc[:, pd.IndexSlice['segment_0', 'target']] += 1
    residuals_df.loc[:, pd.IndexSlice['segment_1', 'target']] -= 1
    return (residuals_df, forecast_df, t)

@pytest.mark.parametrize('detrend_model', (TheilSenRegressor(), LinearRegression()))
def test_plot_bin_seg(example_tsdf, detrend_model):
    """ŪƉɪơ͚  q̔±ʹ Ũ  ˆ  Ƈ       4    """
    plot_trend(ts=example_tsdf, trend_transform=BinsegTrendTransform(in_column='target', detrend_model=detrend_model))

def test_create_holidays_df_non_day_frequ():
    """¬    """
    classic_df = generate_ar_df(periods=30, start_time='2020-01-01', n_segments=1, freq='H')
    t = TSDataset.to_dataset(classic_df)
    holidays = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2020-01-01']), 'upper_window': 3})
    d = _create_holidays_df(holidays, t.index, as_is=False)
    assert d.sum().sum() == 4

def test_create_holidays_df_lower_window_positive(simple_df):
    holidays = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2020-01-07']), 'lower_window': 1})
    with pytest.raises(valueerror):
        _create_holidays_df(holidays, simple_df.index, as_is=False)
