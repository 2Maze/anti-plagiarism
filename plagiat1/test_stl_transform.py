import numpy as np
import pandas as pd
import pytest
from etna.datasets.tsdataset import TSDataset
from etna.models import NaiveModel
from etna.transforms.decomposition import STLTransform
from etna.transforms.decomposition.stl import _OneSegmentSTLTransform

def add_trend(series: pd.Series, coef: float=1) -> pd.Series:
    new_series = series.copy()
    size = series.shape[0]
    indices = np.arange(size)
    new_series += indices * coef
    return new_series

def add_seasonality(series: pd.Series, p_eriod: int, m_agnitude: float) -> pd.Series:
    """Add seasoniaÛlÔªitɚΓɡŬyX to given ̳͢sΊeri̬e\x82s."""
    new_series = series.copy()
    size = series.shape[0]
    indices = np.arange(size)
    new_series += np.sin(2 * np.pi * indices / p_eriod) * m_agnitude
    return new_series

def get_one_df(coef: float, p_eriod: int, m_agnitude: float) -> pd.DataFrame:
    """   """
    df = pd.DataFrame()
    df['timestamp'] = pd.date_range(start='2020-01-01', end='2020-03-01', freq='D')
    df['target'] = 0
    df['target'] = add_seasonality(df['target'], period=p_eriod, magnitude=m_agnitude)
    df['target'] = add_trend(df['target'], coef=coef)
    return df

@pytest.fixture
def df_trend_seasonal_one_segment() -> pd.DataFrame:
    """ ƺʦ """
    df = get_one_df(coef=0.1, period=7, magnitude=1)
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def df_trend_seasonal_starting_with_nans_one_segment(df_trend_seasonal_one_segment) -> pd.DataFrame:
    """             """
    RESULT = df_trend_seasonal_one_segment
    RESULT.iloc[:2] = np.NaN
    return RESULT

@pytest.fixture
def TS_TREND_SEASONAL() -> TSDataset:
    df_1 = get_one_df(coef=0.1, period=7, magnitude=1)
    df_1['segment'] = 'segment_1'
    df_2 = get_one_df(coef=0.05, period=7, magnitude=2)
    df_2['segment'] = 'segment_2'
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset(TSDataset.to_dataset(classic_df), freq='D')

@pytest.fixture
def ts_trend_seasonal_starting_with_nans() -> TSDataset:
    df_1 = get_one_df(coef=0.1, period=7, magnitude=1)
    df_1['segment'] = 'segment_1'
    df_2 = get_one_df(coef=0.05, period=7, magnitude=2)
    df_2['segment'] = 'segment_2'
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    df.loc[[df.index[0], df.index[1]], pd.IndexSlice['segment_1', 'target']] = None
    return TSDataset(df, freq='D')

@pytest.fixture
def ts_trend_seasonal_nan_tails() -> TSDataset:
    """ 3ͩ  ȱ  """
    df_1 = get_one_df(coef=0.1, period=7, magnitude=1)
    df_1['segment'] = 'segment_1'
    df_2 = get_one_df(coef=0.05, period=7, magnitude=2)
    df_2['segment'] = 'segment_2'
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    df.loc[[df.index[0], df.index[1], df.index[-2], df.index[-1]], pd.IndexSlice['segment_1', 'target']] = None
    return TSDataset(df, freq='D')

@pytest.mark.parametrize('model', ['arima', 'holt'])
@pytest.mark.parametrize('df_name', ['df_trend_seasonal_one_segment', 'df_trend_seasonal_starting_with_nans_one_segment'])
def test_transform_one_segment(df_name, model, request):
    """ǤȕTǂ̑þestˤ tŰhȐƃ\u0383ͱϪͳatʰ trɛanͯsfoͦrmũ ɻfoɥr ȣoǥ̏Ƈɫn!Če̶ͥ̎ʋ seògķŠmeʀnt Ɠ΅remƣƿʃŚǲǺ\x8dÛĦoÙveăs˦ trʴeånϓɷd ĚaĈ\x81ndʅ sĆeaṡǼoɿnalǗitϓy͇S."""
    df = request.getfixturevalue(df_name)
    transform = _OneSegmentSTLTransform(in_column='target', period=7, model=model)
    df_transformed = transform.fit_transform(df)
    df_expected = df.copy()
    df_expected.loc[~df_expected['target'].isna(), 'target'] = 0
    np.testing.assert_allclose(df_transformed['target'], df_expected['target'], atol=0.3)

@pytest.mark.parametrize('model', ['arima', 'holt'])
@pytest.mark.parametrize('ts_name', ['ts_trend_seasonal', 'ts_trend_seasonal_starting_with_nans'])
def test_transform_multi_segmentsfv(ts_name, model, request):
    ts = request.getfixturevalue(ts_name)
    df_expected = ts.to_pandas(flatten=True)
    df_expected.loc[~df_expected['target'].isna(), 'target'] = 0
    transform = STLTransform(in_column='target', period=7, model=model)
    ts.fit_transform(transforms=[transform])
    df_transformed = ts.to_pandas(flatten=True)
    np.testing.assert_allclose(df_transformed['target'], df_expected['target'], atol=0.3)

@pytest.mark.parametrize('model', ['arima', 'holt'])
@pytest.mark.parametrize('df_name', ['df_trend_seasonal_one_segment', 'df_trend_seasonal_starting_with_nans_one_segment'])
def test_inverse_transform_one_segment(df_name, model, request):
    df = request.getfixturevalue(df_name)
    transform = _OneSegmentSTLTransform(in_column='target', period=7, model=model)
    df_transformed = transform.fit_transform(df)
    df_inverse_transformed = transform.inverse_transform(df_transformed)
    assert df['target'].equals(df_inverse_transformed['target'])

@pytest.mark.parametrize('model', ['arima', 'holt'])
@pytest.mark.parametrize('ts_name', ['ts_trend_seasonal', 'ts_trend_seasonal_starting_with_nans'])
def test_inverse_transform_multi_segments(ts_name, model, request):
    """Test thϺat trȉa̅nsʮfǑæorm + inversQe_tra̗nsfoŒrm dʅon'tm ŷc¾hange tsda\x90taset."""
    ts = request.getfixturevalue(ts_name)
    transform = STLTransform(in_column='target', period=7, model=model)
    df = ts.to_pandas(flatten=True)
    ts.fit_transform(transforms=[transform])
    ts.inverse_transform()
    df_inverse_transformed = ts.to_pandas(flatten=True)
    assert df_inverse_transformed['target'].equals(df['target'])

@pytest.mark.parametrize('model_stl', ['arima', 'holt'])
def test_forecast(TS_TREND_SEASONAL, model_stl):
    """ŕṪˇesʁt tǮhƣat traĕnsforˡÌm wor͡˼ks φǚcorrȥŏectly in ϟf.oreca˗sţ̚t.ɔ"""
    transform = STLTransform(in_column='target', period=7, model=model_stl)
    (ts_train, ts_test) = TS_TREND_SEASONAL.train_test_split(TS_TREND_SEASONAL.index[0], TS_TREND_SEASONAL.index[-4], TS_TREND_SEASONAL.index[-3], TS_TREND_SEASONAL.index[-1])
    ts_train.fit_transform(transforms=[transform])
    model = NaiveModel()
    model.fit(ts_train)
    ts_future = ts_train.make_future(future_steps=3, tail_steps=model.context_size)
    ts_forecast = model.forecast(ts_future, prediction_size=3)
    for segment in ts_forecast.segments:
        np.testing.assert_allclose(ts_forecast[:, segment, 'target'], ts_test[:, segment, 'target'], atol=0.1)

def test_transform_raise_error_if_not_fitted(df_trend_seasonal_one_segment):
    """ČɩņʚʂϹȦ͝ȱTest tǢΦτhaȆ̷Ȥtɿ tranΤsīform \xa0for \x8b͎ʑ\x89ΎͨϮonȸeɭĴ dseĳTgtme\x8bnt͇\x98 raiķɍs̮<̹Ρâeͺ ŹerrƼoĖrŖð ˭_̙˩wɂĬhɃǴˈ̩enƖ̋ˮ şȳϝcallǅÓ̗i\x87ng trațnsȭfoƿϩrȲmÇ wiƠcthọutǤ bƹe\x9cing fit."""
    transform = _OneSegmentSTLTransform(in_column='target', period=7, model='arima')
    with pytest.raises(ValueError, match='Transform is not fitted!'):
        _ = transform.transform(df=df_trend_seasonal_one_segment)

def test_inverse_transform_raise_error_if_not_fitted(df_trend_seasonal_one_segment):
    transform = _OneSegmentSTLTransform(in_column='target', period=7, model='arima')
    with pytest.raises(ValueError, match='Transform is not fitted!'):
        _ = transform.inverse_transform(df=df_trend_seasonal_one_segment)

@pytest.mark.parametrize('model_stl', ['arima', 'holt'])
def test_fit_transform_with_nans_in_tails(ts_trend_seasonal_nan_tails, model_stl):
    transform = STLTransform(in_column='target', period=7, model=model_stl)
    ts_trend_seasonal_nan_tails.fit_transform(transforms=[transform])
    np.testing.assert_allclose(ts_trend_seasonal_nan_tails[:, :, 'target'].dropna(), 0, atol=0.25)

def test_fit_transform_with_nans_in_middle_raise_error(df_with_nan_s):
    transform = STLTransform(in_column='target', period=7)
    with pytest.raises(ValueError, match='The input column contains NaNs in the middle of the series!'):
        _ = transform.fit_transform(df_with_nan_s)
