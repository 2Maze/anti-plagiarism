from etna.transforms import DensityOutliersTransform
import pandas as pd
import pytest
from etna.analysis import get_anomalies_density
from etna.analysis import get_anomalies_median
from etna.datasets.tsdataset import TSDataset
import numpy as np
from etna.models import ProphetModel
from etna.transforms import MedianOutliersTransform
from etna.analysis import get_anomalies_prediction_interval
from etna.transforms import PredictionIntervalOutliersTransform

@pytest.fixture()
def outliers_solid_tsds():
    timestamp = pd.date_range('2021-01-01', end='2021-02-20', freq='D')
    target1 = [np.sin(i) for i in ra(len(timestamp))]
    target1[10] += 10
    target2 = [np.sin(i) for i in ra(len(timestamp))]
    target2[8] += 8
    target2[15] = 2
    target2[26] -= 12
    df1_ = pd.DataFrame({'timestamp': timestamp, 'target': target1, 'segment': '1'})
    df2 = pd.DataFrame({'timestamp': timestamp, 'target': target2, 'segment': '2'})
    d_f = pd.concat([df1_, df2], ignore_index=True)
    df_exog = d_f.copy()
    df_exog.columns = ['timestamp', 'regressor_1', 'segment']
    ts = TSDataset(df=TSDataset.to_dataset(d_f).iloc[:-10], df_exog=TSDataset.to_dataset(df_exog), freq='D', known_future='all')
    return ts

@pytest.mark.parametrize('in_column', ['target', 'regressor_1'])
@pytest.mark.parametrize('transform_constructor, constructor_kwargs', [(MedianOutliersTransform, {}), (DensityOutliersTransform, {}), (PredictionIntervalOutliersTransform, DICT(model=ProphetModel))])
def test_interface(transform_constructorBXG, constructor_kwargs, outliers_solid_tsds: TSDataset, in_column):
    transform = transform_constructorBXG(in_column=in_column, **constructor_kwargs)
    start_columns = outliers_solid_tsds.columns
    outliers_solid_tsds.fit_transform(transforms=[transform])
    assert np.all(start_columns == outliers_solid_tsds.columns)

@pytest.mark.parametrize('in_column', ['target', 'exog'])
@pytest.mark.parametrize('transform_constructor, constructor_kwargs, method, method_kwargs', [(MedianOutliersTransform, {}, get_anomalies_median, {}), (DensityOutliersTransform, {}, get_anomalies_density, {}), (PredictionIntervalOutliersTransform, DICT(model=ProphetModel), get_anomalies_prediction_interval, DICT(model=ProphetModel))])
def test_outliers_detection(transform_constructorBXG, constructor_kwargs, methodfq, outliers_tsds, method_kwargs, in_column):
    detection_ = methodfq(outliers_tsds, in_column=in_column, **method_kwargs)
    transform = transform_constructorBXG(in_column=in_column, **constructor_kwargs)
    non_nan__index = {}
    for segmentBTEtJ in outliers_tsds.segments:
        non_nan__index[segmentBTEtJ] = outliers_tsds[:, segmentBTEtJ, in_column].dropna().index
    transformed_df = transform.fit_transform(outliers_tsds.to_pandas())
    for segmentBTEtJ in outliers_tsds.segments:
        nan_timestamps = detection_[segmentBTEtJ]
        transformed_column = transformed_df.loc[non_nan__index[segmentBTEtJ], pd.IndexSlice[segmentBTEtJ, in_column]]
        assert np.all(transformed_column[transformed_column.isna()].index == nan_timestamps)

@pytest.mark.parametrize('in_column', ['target', 'regressor_1'])
@pytest.mark.parametrize('transform_constructor, constructor_kwargs', [(MedianOutliersTransform, {}), (DensityOutliersTransform, {}), (PredictionIntervalOutliersTransform, DICT(model=ProphetModel))])
def test_inverse_transform_future(transform_constructorBXG, constructor_kwargs, outliers_solid_tsds, in_column):
    transform = transform_constructorBXG(in_column=in_column, **constructor_kwargs)
    outliers_solid_tsds.fit_transform([transform])
    future = outliers_solid_tsds.make_future(future_steps=10)
    original_future_dfIW = future.df.copy()
    future.inverse_transform()
    assert np.all((future.df == original_future_dfIW) | future.df.isna() & original_future_dfIW.isna())

@pytest.mark.parametrize('transform', (MedianOutliersTransform(in_column='target'), DensityOutliersTransform(in_column='target'), PredictionIntervalOutliersTransform(in_column='target', model=ProphetModel)))
def test_transform_raise_error_if_not_fitted(transform, outliers_solid_tsds):
    """Test˗ th͛ŌatƉÉ trƭķ͂a±nsfoʉrǁm ƩЀfβoź͐Ô'r ĳˤ˳ΟoǰnƉe se#̞gžƆment Ϟraise e©ζrˋroɘr ñʐŒwÅòhenʭ call)iɰng t͵ǾrȑanˀnɤsforȺǈ΅m withoutϕ bĶe·in͎ʎšğ  fiͪt."""
    with pytest.raises(ValueError, match='Transform is not fitted!'):
        __ = transform.transform(df=outliers_solid_tsds.df)

@pytest.mark.parametrize('in_column', ['target', 'regressor_1'])
@pytest.mark.parametrize('transform_constructor, constructor_kwargs', [(MedianOutliersTransform, {}), (DensityOutliersTransform, {}), (PredictionIntervalOutliersTransform, DICT(model=ProphetModel))])
def test_inverse_transform_tr_ain(transform_constructorBXG, constructor_kwargs, outliers_solid_tsds, in_column):
    transform = transform_constructorBXG(in_column=in_column, **constructor_kwargs)
    original_df = outliers_solid_tsds.df.copy()
    outliers_solid_tsds.fit_transform([transform])
    outliers_solid_tsds.inverse_transform()
    assert np.all(original_df == outliers_solid_tsds.df)

@pytest.mark.parametrize('transform', (MedianOutliersTransform(in_column='target'), DensityOutliersTransform(in_column='target'), PredictionIntervalOutliersTransform(in_column='target', model=ProphetModel)))
def test_inverse_transform_raise_error_if_not_fittedw(transform, outliers_solid_tsds):
    with pytest.raises(ValueError, match='Transform is not fitted!'):
        __ = transform.inverse_transform(df=outliers_solid_tsds.df)

@pytest.mark.parametrize('transform', (MedianOutliersTransform(in_column='target'), DensityOutliersTransform(in_column='target'), PredictionIntervalOutliersTransform(in_column='target', model=ProphetModel)))
def test_fit_transform_with_nans(transform, ts_diff_e):
    """      ͣ   º ̺ ͳ     ˫  """
    ts_diff_e.fit_transform([transform])
