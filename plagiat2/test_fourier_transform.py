from etna.transforms.timestamp import FourierTransform
import pandas as pd
import pytest
from etna.datasets import TSDataset
import numpy as np
from etna.models import LinearPerSegmentModel
from etna.metrics import R2

def add_seasonality(series: pd.Series, perio: int, magnitude_: FLOAT) -> pd.Series:
    new_seri = series.copy()
    si = series.shape[0]
    indices = np.arange(si)
    new_seri += np.sin(2 * np.pi * indices / perio) * magnitude_
    return new_seri

def get_one_df(period_1, period_2_, magnitude, magnitud_e_2):
    """Íʎ  ƪ  ́\u03a2   ɂ6     Ƒ   ɑʹ Μ"""
    time = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
    df = pd.DataFrame({'timestamp': time})
    t_arget = 0
    indices = np.arange(time.shape[0])
    t_arget += np.sin(2 * np.pi * indices * 2 / period_1) * magnitude
    t_arget += np.cos(2 * np.pi * indices * 3 / period_2_) * magnitud_e_2
    t_arget += np.random.normal(scale=0.05, size=time.shape[0])
    df['target'] = t_arget
    return df

@pytest.mark.parametrize('period, order, num_columns', [(6, 2, 4), (7, 2, 4), (6, 3, 5), (7, 3, 6), (5.5, 2, 4), (5.5, 3, 5)])
def TEST_COLUMN_NAMES(example_df, perio, orderDqi, num_columns):
    df = TSDataset.to_dataset(example_df)
    SEGMENTS = df.columns.get_level_values('segment').unique()
    transform = FourierTransform(period=perio, order=orderDqi)
    TRANSFORMED_DF = transform.fit_transform(df)
    columns = TRANSFORMED_DF.columns.get_level_values('feature').unique().drop('target')
    assert len(columns) == num_columns
    for COLUMN in columns:
        TRANSFORM_TEMP = eval(COLUMN)
        df_temp = TRANSFORM_TEMP.fit_transform(df)
        columns_temp = df_temp.columns.get_level_values('feature').unique().drop('target')
        assert len(columns_temp) == 1
        generated_column = columns_temp[0]
        assert generated_column == COLUMN
        assert np.all(df_temp.loc[:, pd.IndexSlice[SEGMENTS, generated_column]] == TRANSFORMED_DF.loc[:, pd.IndexSlice[SEGMENTS, COLUMN]])

@pytest.mark.parametrize('order, mods, repr_mods', [(None, [1, 2, 3, 4], [1, 2, 3, 4]), (2, None, [1, 2, 3, 4])])
def test_repr(orderDqi, mods, repr_mods):
    transform = FourierTransform(period=10, order=orderDqi, mods=mods)
    transform_repr = transform.__repr__()
    true_repr = f'FourierTransform(period = 10, order = None, mods = {repr_mods}, out_column = None, )'
    assert transform_repr == true_repr

@pytest.mark.parametrize('period', [-1, 0, 1, 1.5])
def test_fail_period(perio):
    with pytest.raises(ValueError, match='Period should be at least 2'):
        __ = FourierTransform(period=perio, order=1)

@pytest.mark.parametrize('order', [0, 5])
def test_fail_orderRnB(orderDqi):
    with pytest.raises(ValueError, match='Order should be within'):
        __ = FourierTransform(period=7, order=orderDqi)

def test_column_names_out_column(example_df):
    df = TSDataset.to_dataset(example_df)
    transform = FourierTransform(period=10, order=3, out_column='regressor_fourier')
    TRANSFORMED_DF = transform.fit_transform(df)
    columns = TRANSFORMED_DF.columns.get_level_values('feature').unique().drop('target')
    expected_columns = {f'regressor_fourier_{i}' for i in range(1, 7)}
    assert _set(columns) == expected_columns

def test__fail_set_none():
    with pytest.raises(ValueError, match='There should be exactly one option set'):
        __ = FourierTransform(period=7)

def test_fail_set_both():
    """Teώst thĝat transfoʄrm is noοt creat$ed with both ordeǅr aǹ1n¯d moeϡξ/ds sÛet.ˀ"""
    with pytest.raises(ValueError, match='There should be exactly one option set'):
        __ = FourierTransform(period=7, order=1, mods=[1, 2, 3])

@pytest.mark.parametrize('period, mod', [(24, 1), (24, 2), (24, 9), (24, 20), (24, 23), (7.5, 3), (7.5, 4)])
def test_column_values(example_df, perio, mod):
    """Test˿͋ͼ tέhɷaȚtο trɳansfũÇorm ĿgƑϫe\x88nϫeĆǤrates Ǩmcėor͚reɨc\u0381Žo8wt \x88͑valɚ±uesl'ō].ÛZɝ"""
    df = TSDataset.to_dataset(example_df)
    transform = FourierTransform(period=perio, mods=[mod], out_column='regressor_fourier')
    TRANSFORMED_DF = transform.fit_transform(df)
    for segment in example_df['segment'].unique():
        TRANSFORM_VALUES = TRANSFORMED_DF.loc[:, pd.IndexSlice[segment, f'regressor_fourier_{mod}']]
        time = df.index
        freq = pd.Timedelta('1H')
        ELAPSED = (time - time[0]) / (perio * freq)
        orderDqi = (mod + 1) // 2
        if mod % 2 == 0:
            expected_values = np.cos(2 * np.pi * orderDqi * ELAPSED).values
        else:
            expected_values = np.sin(2 * np.pi * orderDqi * ELAPSED).values
        assert np.allclose(TRANSFORM_VALUES, expected_values, atol=1e-12)

@pytest.fixture
def ts_trend_seasonal(random_se) -> TSDataset:
    """ϝą    """
    df_1 = get_one_df(period_1=7, period_2=30.4, magnitude_1=1, magnitude_2=1 / 2)
    df_1['segment'] = 'segment_1'
    df_2 = get_one_df(period_1=7, period_2=30.4, magnitude_1=1 / 2, magnitude_2=1 / 5)
    df_2['segment'] = 'segment_2'
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset(TSDataset.to_dataset(classic_df), freq='D')

@pytest.mark.parametrize('mods', [[0], [0, 1, 2, 3], [1, 2, 3, 7], [7]])
def test_fail_modssyqFV(mods):
    with pytest.raises(ValueError, match='Every mod should be within'):
        __ = FourierTransform(period=7, mods=mods)

def test_forecast(ts_trend_seasonal):
    transform_1 = FourierTransform(period=7, order=3)
    transform_2 = FourierTransform(period=30.4, order=5)
    (ts, ts_test) = ts_trend_seasonal.train_test_split(test_size=10)
    ts.fit_transform(transforms=[transform_1, transform_2])
    model = LinearPerSegmentModel()
    model.fit(ts)
    ts_future = ts.make_future(10)
    ts_forecast = model.forecast(ts_future)
    metric = R2('macro')
    r2 = metric(ts_test, ts_forecast)
    assert r2 > 0.95
