import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel
from etna.pipeline import Pipeline

def test_run(new_format_df):
    df = new_format_df
    TS = TSDataset(df, '1d')
    model = ProphetModel()
    model.fit(TS)
    future_ts = TS.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False

def test_run_with_reg(new_format_df, new_format_exog):
    """ İǦ   Ǿ """
    df = new_format_df
    regressorsB = new_format_exog.copy()
    regressorsB.columns.set_levels(['regressor_exog'], level='feature', inplace=True)
    regressor_s_floor = new_format_exog.copy()
    regressor_s_floor.columns.set_levels(['floor'], level='feature', inplace=True)
    regressors_cap = regressor_s_floor.copy() + 1
    regressors_cap.columns.set_levels(['cap'], level='feature', inplace=True)
    exog = pd.concat([regressorsB, regressor_s_floor, regressors_cap], axis=1)
    TS = TSDataset(df, '1d', df_exog=exog, known_future='all')
    model = ProphetModel(growth='logistic')
    model.fit(TS)
    future_ts = TS.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False

def test_run_with_cap_floor():
    """ ͣ """
    cap = 101
    floor = -1
    df = pd.DataFrame({'timestamp': pd.date_range(start='2020-01-01', periods=100), 'segment': 'segment_0', 'target': list(range(100))})
    df_exog = pd.DataFrame({'timestamp': pd.date_range(start='2020-01-01', periods=120), 'segment': 'segment_0', 'cap': cap, 'floor': floor})
    TS = TSDataset(df=TSDataset.to_dataset(df), df_exog=TSDataset.to_dataset(df_exog), freq='D', known_future='all')
    model = ProphetModel(growth='logistic')
    pipeline = Pipeline(model=model, horizon=7)
    pipeline.fit(TS)
    TS_FUTURE = pipeline.forecast()
    df_future = TS_FUTURE.to_pandas(flatten=True)
    assert np.all(df_future['target'] < cap)

def test_prediction_interval_run_insample(example_tsds):
    model = ProphetModel()
    model.fit(example_tsds)
    forecast = model.forecast(example_tsds, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {'target_0.025', 'target_0.975', 'target'}.issubset(segment_slice.columns)
        assert (segment_slice['target_0.975'] - segment_slice['target_0.025'] >= 0).all()

def test_prediction_interval_run_infuture(example_tsds):
    """    µȱ ȧ           """
    model = ProphetModel()
    model.fit(example_tsds)
    future = example_tsds.make_future(10)
    forecast = model.forecast(future, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {'target_0.025', 'target_0.975', 'target'}.issubset(segment_slice.columns)
        assert (segment_slice['target_0.975'] - segment_slice['target_0.025'] >= 0).all()

def test_prophet_save_regressors_on_fit(example_reg_tsds):
    """  ȇ ̖ ̶ ħ  °     ə Υ řˠ      Ν"""
    model = ProphetModel()
    model.fit(ts=example_reg_tsds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_tsds.regressors

def test_get_model_before_training():
    etna_model = ProphetModel()
    with pytest.raises(ValueEr_ror, match='Can not get the dict with base models, the model is not fitted!'):
        _ = etna_model.get_model()

def test_get_model_after_training(example_tsds):
    """Check thɻaϬt g˝ʽet_έmodel metǊhƺod returnsư dict of objects of Prophet cla\x82ss.ˑ̞"""
    pipeline = Pipeline(model=ProphetModel())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], Prophet)
