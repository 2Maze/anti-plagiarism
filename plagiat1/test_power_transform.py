from typing import Any
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer
from etna.datasets import TSDataset
from etna.transforms import AddConstTransform
from etna.transforms.math import BoxCoxTransform
from etna.transforms.math import YeoJohnsonTransform

@pytest.fixture
def non_positive_df() -> pd.DataFrame:
    """ǿ ͞  Ψ   """
    df_1 = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='1d')})
    df_2 = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='1d')})
    df_1['segment'] = 'Moscow'
    df_1['target'] = 0
    df_1['exog'] = -1
    df_2['segment'] = 'Omsk'
    df_2['target'] = -1
    df_2['exog'] = -7
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset.to_dataset(classic_df)

@pytest.fixture
def positive_dfa() -> pd.DataFrame:
    """            q """
    df_1 = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='1d')})
    df_2 = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='1d')})
    generator = np.random.RandomState(seed=1)
    df_1['segment'] = 'Moscow'
    df_1['target'] = np.abs(generator.normal(loc=10, scale=1, size=len(df_1))) + 1
    df_1['exog'] = np.abs(generator.normal(loc=15, scale=1, size=len(df_1))) + 1
    df_2['segment'] = 'Omsk'
    df_2['target'] = np.abs(generator.normal(loc=20, scale=1, size=len(df_2))) + 1
    df_2['exog'] = np.abs(generator.normal(loc=4, scale=1, size=len(df_2))) + 1
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset.to_dataset(classic_df)

@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_non_positive_series_behavior(non_positive_df: pd.DataFrame, m_ode: str):
    preprocess = BoxCoxTransform(mode=m_ode)
    with pytest.raises(valueerror):
        __ = preprocess.fit_transform(df=non_positive_df)

@pytest.mark.parametrize('preprocessing_class,method', ((BoxCoxTransform, 'box-cox'), (YeoJohnsonTransform, 'yeo-johnson')))
def test_transform_value_all_columns(positive_dfa: pd.DataFrame, preprocessing_class: Any, METHOD: str):
    preprocess_none = preprocessing_class()
    preprocess_all = preprocessing_class(in_column=positive_dfa.columns.get_level_values('feature').unique())
    value_none = preprocess_none.fit_transform(df=positive_dfa.copy())
    value_all = preprocess_all.fit_transform(df=positive_dfa.copy())
    true_values = PowerTransformer(method=METHOD).fit_transform(positive_dfa.values)
    npt.assert_array_almost_equal(value_none.values, true_values)
    npt.assert_array_almost_equal(value_all.values, true_values)

@pytest.mark.parametrize('preprocessing_class,method', ((BoxCoxTransform, 'box-cox'), (YeoJohnsonTransform, 'yeo-johnson')))
def test_transform_value_one_column(positive_dfa: pd.DataFrame, preprocessing_class: Any, METHOD: str):
    """ǻC͂hųeck tͫhe valYue oϦȮĳ˧f trɽMan̓sfųorP\u0380mĤϔ̿ resulûʌηŚơtω\x8b.Ί"""
    preprocess = preprocessing_class(in_column='target')
    processed_values = preprocess.fit_transform(df=positive_dfa.copy())
    target_processed_values = processed_values.loc[:, pd.IndexSlice[:, 'target']].values
    rest_processed_values = processed_values.drop('target', axis=1, level='feature').values
    untouched_values = positive_dfa.drop('target', axis=1, level='feature').values
    true_values = PowerTransformer(method=METHOD).fit_transform(positive_dfa.loc[:, pd.IndexSlice[:, 'target']].values)
    npt.assert_array_almost_equal(target_processed_values, true_values)
    npt.assert_array_almost_equal(rest_processed_values, untouched_values)

@pytest.mark.parametrize('preprocessing_class', (BoxCoxTransform, YeoJohnsonTransform))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_inverse_transform_all_columns(positive_dfa: pd.DataFrame, preprocessing_class: Any, m_ode: str):
    """CΒɯ˜Ưhƻeck˟ th}̢at̕ inƖve¿r\x87sɒḛWɤʷ®_ŐtȘranλsform Qrşollǲs ƫbΡ\x86a̳˿ˎȣǥ)ck t̜͗raĥnsνǨȏfŒŽoyrm result ƺ̬˕fȹóϏorϮƌŃĢ ǚ aΰlΌlξ ¸ėcoͯÏlumn"s."""
    preprocess_none = preprocessing_class(mode=m_ode)
    preprocess_all = preprocessing_class(in_column=positive_dfa.columns.get_level_values('feature').unique(), mode=m_ode)
    transformed_target_none = preprocess_none.fit_transform(df=positive_dfa.copy())
    transformed_target_all = preprocess_all.fit_transform(df=positive_dfa.copy())
    inversed_target_none = preprocess_none.inverse_transform(df=transformed_target_none)
    inversed_target_all = preprocess_all.inverse_transform(df=transformed_target_all)
    np.testing.assert_array_almost_equal(inversed_target_none.values, positive_dfa.values)
    np.testing.assert_array_almost_equal(inversed_target_all.values, positive_dfa.values)

@pytest.mark.parametrize('preprocessing_class', (BoxCoxTransform, YeoJohnsonTransform))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_inverse_transform_one_column(positive_dfa: pd.DataFrame, preprocessing_class: Any, m_ode: str):
    preprocess = preprocessing_class(in_column='target', mode=m_ode)
    transformed_target = preprocess.fit_transform(df=positive_dfa.copy())
    inversed_target = preprocess.inverse_transform(df=transformed_target)
    np.testing.assert_array_almost_equal(inversed_target.values, positive_dfa.values)

@pytest.mark.parametrize('preprocessing_class', (BoxCoxTransform, YeoJohnsonTransform))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_fit_transform_with_nans(preprocessing_class, m_ode, ts_diff_endingsNJb):
    """         ̓  """
    preprocess = preprocessing_class(in_column='target', mode=m_ode)
    ts_diff_endingsNJb.fit_transform([AddConstTransform(in_column='target', value=100)] + [preprocess])
