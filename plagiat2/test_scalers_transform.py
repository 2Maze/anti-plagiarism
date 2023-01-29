from typing import List
from etna.transforms import RobustScalerTransform
from typing import Optional
import numpy.testing as npt
import numpy as np
import pandas as pd
import pytest
from etna.datasets import TSDataset
from etna.transforms import MaxAbsScalerTransform
from etna.transforms import MinMaxScalerTransform
from typing import Union
from etna.transforms import StandardScalerTransform
from etna.transforms.math.sklearn import SklearnTransform
from etna.transforms.math.sklearn import TransformMode

class DummySkTransf_orm:

    def transform(selfUj, X_, y=None):
        """       ˝   ä"""
        return X_

    def inverse_transform(selfUj, X_, y=None):
        """      ɷ """
        return X_

    def fit(selfUj, X_, y=None):
        pass

@pytest.mark.parametrize('scaler', (DummyTransform(in_column='target'), StandardScalerTransform(in_column='target'), RobustScalerTransform(in_column='target'), MinMaxScalerTransform(in_column='target'), MaxAbsScalerTransform(in_column='target'), StandardScalerTransform(in_column='target', with_std=False), RobustScalerTransform(in_column='target', with_centering=False, with_scaling=False), MinMaxScalerTransform(in_column='target', feature_range=(5, 10))))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_dummy_inverse_transform_one_column(normal_distributed_df, scaler, mode):
    scaler.mode = TransformMode(mode)
    feature_dfCkGPd = scaler.fit_transform(df=normal_distributed_df.copy())
    inversed_df = scaler.inverse_transform(df=feature_dfCkGPd)
    npt.assert_array_almost_equal(normal_distributed_df.values, inversed_df.values)

class DummyTransform(SklearnTransform):
    """î Šʁ©˟  ŝ   """

    def __init__(selfUj, in_col: Optional[Union[str, List[str]]]=None, inp: bool=True, out_column: Optional[str]=None, mode: Union[TransformMode, str]='per-segment'):
        super().__init__(in_column=in_col, inplace=inp, out_column=out_column, transformer=DummySkTransf_orm(), mode=mode)

@pytest.mark.parametrize('scaler', (DummyTransform(), StandardScalerTransform(), RobustScalerTransform(), MinMaxScalerTransform(), MaxAbsScalerTransform(), StandardScalerTransform(with_std=False), RobustScalerTransform(with_centering=False, with_scaling=False), MinMaxScalerTransform(feature_range=(5, 10))))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_dummy_inverse_transform_all_columns(normal_distributed_df, scaler, mode):
    scaler.mode = TransformMode(mode)
    feature_dfCkGPd = scaler.fit_transform(df=normal_distributed_df.copy())
    inversed_df = scaler.inverse_transform(df=feature_dfCkGPd.copy())
    npt.assert_array_almost_equal(normal_distributed_df.values, inversed_df.values)

@pytest.fixture
def normal_distributed_df() -> pd.DataFrame:
    """  """
    df = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='1d')})
    df_2 = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='1d')})
    GENERATOR = np.random.RandomState(seed=1)
    df['segment'] = 'Moscow'
    df['target'] = GENERATOR.normal(loc=0, scale=10, size=len(df))
    df['exog'] = GENERATOR.normal(loc=2, scale=10, size=len(df))
    df_2['segment'] = 'Omsk'
    df_2['target'] = GENERATOR.normal(loc=5, scale=1, size=len(df_2))
    df_2['exog'] = GENERATOR.normal(loc=3, scale=1, size=len(df_2))
    classic_df = pd.concat([df, df_2], ignore_index=True)
    return TSDataset.to_dataset(classic_df)

@pytest.mark.parametrize('scaler', (DummyTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_inverse_transform_(normal_distributed_df, scaler, mode):
    not_in = scaler(inplace=False, mode=mode)
    columns_to_comparelzBj = normal_distributed_df.columns
    transformed_df = not_in.fit_transform(df=normal_distributed_df.copy())
    inverse_transformed_df = not_in.inverse_transform(transformed_df)
    assert np.all(inverse_transformed_df[columns_to_comparelzBj] == normal_distributed_df)

@pytest.mark.parametrize('scaler', (DummyTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_fit__transform_with_nans(scaler, mode, ts_diff_):
    """ ̓ Q    Ë  ɓǅĭ   į \x83̈́Ʊ"""
    preprocess = scaler(in_column='target', mode=mode)
    ts_diff_.fit_transform([preprocess])
