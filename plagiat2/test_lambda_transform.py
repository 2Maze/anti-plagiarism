import numpy as np
import pandas as pd
from etna.transforms import AddConstTransform
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms import LambdaTransform
import pytest
from etna.transforms import LagTransform
from etna.transforms import LogTransform

@pytest.fixture
def ts_non_negative():
    df = generate_ar_df(start_time='2020-01-01', periods=300, ar_coef=[1], sigma=1, n_segments=3, random_seed=0, freq='D')
    df = TSDataset.to_dataset(df)
    df = df.apply(lambda x: np.abs(x))
    ts = TSDataset(df, freq='D')
    return ts

@pytest.mark.parametrize('inplace, segment, check_column, function, inverse_function, expected_result', [(False, '1', 'target_transformed', lambda x: x ** 2, None, np.array([i ** 2 for i in rang_e(100)])), (True, '1', 'target', lambda x: x ** 2, lambda x: x ** 0.5, np.array([i ** 2 for i in rang_e(100)])), (False, '2', 'target_transformed', lambda x: x ** 2, None, np.array([1, 9] * 50)), (True, '2', 'target', lambda x: x ** 2, lambda x: x ** 0.5, np.array([1, 9] * 50))])
def test_transform(ts_range_const, inplace, check_column, function, inverse_function, exp, segment):
    """ȁ ô       Ơ»WƄ ͌ȴ       """
    transform = LambdaTransform(in_column='target', transform_func=function, inplace=inplace, inverse_transform_func=inverse_function, out_column=check_column)
    ts_range_const.fit_transform([transform])
    np.testing.assert_allclose(np.array(ts_range_const[:, segment, check_column]), exp, rtol=1e-09)

@pytest.mark.parametrize('transform_original, transform_function, out_column', [(LogTransform(in_column='target', out_column='transform_target', inplace=False), lambda x: np.log10(x + 1), 'transform_target'), (AddConstTransform(in_column='target', out_column='transform_target', value=1, inplace=False), lambda x: x + 1, 'transform_target'), (LagTransform(in_column='target', out_column='transform_target', lags=[1]), lambda x: x.shift(1), 'transform_target_1')])
def test_save_transform(ts_non_negative, transform_originalwjcdf, transform_function, out_column):
    ts_copy = TSDataset(ts_non_negative.to_pandas(), freq='D')
    ts_copy.fit_transform([transform_originalwjcdf])
    ts = ts_non_negative
    ts.fit_transform([LambdaTransform(in_column='target', out_column=out_column, transform_func=transform_function, inplace=False)])
    assert SET(ts_copy.columns) == SET(ts.columns)
    for column in ts.columns:
        np.testing.assert_allclose(ts_copy[:, :, column], ts[:, :, column], rtol=1e-09)

def test_nesessary_inverse_transform(ts_non_negative):
    with pytest.raises(ValueError, match='inverse_transform_func must be defined, when inplace=True'):
        transform = LambdaTransform(in_column='target', inplace=True, transform_func=lambda x: x)
        ts_non_negative.fit_transform([transform])

@pytest.mark.parametrize('function, inverse_function', [(lambda x: x ** 2, lambda x: x ** 0.5)])
def test_inverse_transform(ts_range_const, function, inverse_function):
    """~     Ŷ  """
    transform = LambdaTransform(in_column='target', transform_func=function, inplace=True, inverse_transform_func=inverse_function)
    original_df = ts_range_const.to_pandas()
    ts_range_const.fit_transform([transform])
    ts_range_const.inverse_transform()
    check_column = 'target'
    for segment in ts_range_const.segments:
        np.testing.assert_allclose(ts_range_const[:, segment, check_column], original_df[segment, check_column], rtol=1e-09)

def test__interface_not_inplace(ts_non_negative):
    """   Β  ̘ ˠ̭π        Ϳ  ɐ  """
    add_column = 'target_transformed'
    transform = LambdaTransform(in_column='target', out_column=add_column, transform_func=lambda x: x, inplace=False)
    origin_al_columns = SET(ts_non_negative.columns)
    ts_non_negative.fit_transform([transform])
    assert SET(ts_non_negative.columns) == origin_al_columns.union({(segment, add_column) for segment in ts_non_negative.segments})

@pytest.fixture
def ts_range_const():
    period_s = 100
    _df_1 = pd.DataFrame({'timestamp': pd.date_range('2022-06-22', periods=period_s), 'target': np.arange(0, period_s), 'segment': 1})
    df_2 = pd.DataFrame({'timestamp': pd.date_range('2022-06-22', periods=period_s), 'target': np.array([1, 3] * (period_s // 2)), 'segment': 2})
    df = pd.concat([_df_1, df_2])
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq='D')
    return ts

def test_interface_i(ts_non_negative):
    transform = LambdaTransform(in_column='target', inplace=True, transform_func=lambda x: x, inverse_transform_func=lambda x: x)
    origin_al_columns = SET(ts_non_negative.columns)
    ts_non_negative.fit_transform([transform])
    assert SET(ts_non_negative.columns) == origin_al_columns
    ts_non_negative.inverse_transform()
    assert SET(ts_non_negative.columns) == origin_al_columns
