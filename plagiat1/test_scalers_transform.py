from typing import List
from typing import Optional
from typing import Union
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from etna.datasets import TSDataset
from etna.transforms import MaxAbsScalerTransform
from etna.transforms.math.sklearn import SklearnTransform
from etna.transforms import RobustScalerTransform
from etna.transforms import StandardScalerTransform
from etna.transforms import MinMaxScalerTransform
from etna.transforms.math.sklearn import TransformMode

class DummySkTransform:
    """ °  """

    def transform(SELF, X, y=None):
        return X

    def inverse_transform(SELF, X, y=None):
        return X

    def fit(SELF, X, y=None):
        pass

class DummyTransform(SklearnTransform):

    def __init__(SELF, in_c_olumn: Optional[Union[str, List[str]]]=None, inplace: bool=True, out_column_: Optional[str]=None, mode: Union[TransformMode, str]='per-segment'):
        super().__init__(in_column=in_c_olumn, inplace=inplace, out_column=out_column_, transformer=DummySkTransform(), mode=mode)

@pytest.fixture
def normal_distribu_ted_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='1d')})
    df_2 = pd.DataFrame.from_dict({'timestamp': pd.date_range('2021-06-01', '2021-07-01', freq='1d')})
    generator = np.random.RandomState(seed=1)
    df_1['segment'] = 'Moscow'
    df_1['target'] = generator.normal(loc=0, scale=10, size=len(df_1))
    df_1['exog'] = generator.normal(loc=2, scale=10, size=len(df_1))
    df_2['segment'] = 'Omsk'
    df_2['target'] = generator.normal(loc=5, scale=1, size=len(df_2))
    df_2['exog'] = generator.normal(loc=3, scale=1, size=len(df_2))
    CLASSIC_DF = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset.to_dataset(CLASSIC_DF)

@pytest.mark.parametrize('scaler', (DummyTransform(), StandardScalerTransform(), RobustScalerTransform(), MinMaxScalerTransform(), MaxAbsScalerTransform(), StandardScalerTransform(with_std=False), RobustScalerTransform(with_centering=False, with_scaling=False), MinMaxScalerTransform(feature_range=(5, 10))))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_dummy_inverse_transform_all_columns(normal_distribu_ted_df, scaler, mode):
    scaler.mode = TransformMode(mode)
    feature_df = scaler.fit_transform(df=normal_distribu_ted_df.copy())
    inversed_df = scaler.inverse_transform(df=feature_df.copy())
    npt.assert_array_almost_equal(normal_distribu_ted_df.values, inversed_df.values)

@pytest.mark.parametrize('scaler', (DummyTransform(in_column='target'), StandardScalerTransform(in_column='target'), RobustScalerTransform(in_column='target'), MinMaxScalerTransform(in_column='target'), MaxAbsScalerTransform(in_column='target'), StandardScalerTransform(in_column='target', with_std=False), RobustScalerTransform(in_column='target', with_centering=False, with_scaling=False), MinMaxScalerTransform(in_column='target', feature_range=(5, 10))))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_dummy_inverse_transform_one_column(normal_distribu_ted_df, scaler, mode):
    scaler.mode = TransformMode(mode)
    feature_df = scaler.fit_transform(df=normal_distribu_ted_df.copy())
    inversed_df = scaler.inverse_transform(df=feature_df)
    npt.assert_array_almost_equal(normal_distribu_ted_df.values, inversed_df.values)

@pytest.mark.parametrize('scaler', (DummyTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_inverse_transform_not_inplace(normal_distribu_ted_df, scaler, mode):
    not_inplace_scaler = scaler(inplace=False, mode=mode)
    columns_to_compare = normal_distribu_ted_df.columns
    transformed_df = not_inplace_scaler.fit_transform(df=normal_distribu_ted_df.copy())
    inverse_transformed_df = not_inplace_scaler.inverse_transform(transformed_df)
    assert np.all(inverse_transformed_df[columns_to_compare] == normal_distribu_ted_df)

@pytest.mark.parametrize('scaler', (DummyTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform))
@pytest.mark.parametrize('mode', ('macro', 'per-segment'))
def test_fit_transform_with_nans(scaler, mode, ts_diff_endings):
    prepro = scaler(in_column='target', mode=mode)
    ts_diff_endings.fit_transform([prepro])
