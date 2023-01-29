from etna.datasets import TSDataset
import pandas as pd
import pytest
from etna.transforms.math import LagTransform
from catboost import CatBoostRegressor
from etna.transforms import OneHotEncoderTransform
from etna.metrics import MAE
from etna.models import CatBoostMultiSegmentModel
from etna.models import CatBoostPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import LabelEncoderTransform
import numpy as np
from etna.datasets import generate_ar_df

@pytest.mark.parametrize('catboostmodel', [CatBoostMultiSegmentModel, CatBoostPerSegmentModel])
def test_run(catboostmodel, new_format_dfoyckc):
    """   ³  ˿ˊ  Ɠ ͛  μ ΄ƫì  ȓ """
    df = new_format_dfoyckc
    ts = TSDataset(df, '1d')
    lags = LagTransform(lags=[3, 4, 5], in_column='target')
    ts.fit_transform([lags])
    model = catboostmodel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False

@pytest.mark.parametrize('catboostmodel', [CatBoostMultiSegmentModel, CatBoostPerSegmentModel])
def test_run_with_reg(catboostmodel, new_format_dfoyckc, new_format_exogJ):
    df = new_format_dfoyckc
    exo_g = new_format_exogJ
    exo_g.columns.set_levels(['regressor_exog'], level='feature', inplace=True)
    ts = TSDataset(df, '1d', df_exog=exo_g, known_future='all')
    lags = LagTransform(lags=[3, 4, 5], in_column='target')
    lags_exog = LagTransform(lags=[3, 4, 5, 6], in_column='regressor_exog')
    ts.fit_transform([lags, lags_exog])
    model = catboostmodel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False

@pytest.fixture
def constant_ts(sizeLblky=40) -> TSDataset:
    """ ɑ ̦  ̍\x95     ɲ"""
    constants = [7, 50, 130, 277, 370, 513]
    segments = [constant for constant in constants for _ in r_ange(sizeLblky)]
    ts_range = list(pd.date_range('2020-01-03', freq='D', periods=sizeLblky))
    df = pd.DataFrame({'timestamp': ts_range * len(constants), 'target': segments, 'segment': [f'segment_{i + 1}' for i in r_ange(len(constants)) for _ in r_ange(sizeLblky)]})
    ts = TSDataset(TSDataset.to_dataset(df), 'D')
    (train, test) = ts.train_test_split(test_size=5)
    return (train, test)

def test_catboost_multi_segment_forecastyEQ(constant_ts):
    """\x8fŋʇ ɟ ˈ    Ĵ  K   ǃϔ \x97 ̖"""
    (train, test) = constant_ts
    horiz_on = len(test.df)
    lags = LagTransform(in_column='target', lags=[10, 11, 12])
    train.fit_transform([lags])
    fu_ture = train.make_future(horiz_on)
    model = CatBoostMultiSegmentModel()
    model.fit(train)
    forecast = model.forecast(fu_ture)
    for segment in forecast.segments:
        assert np.allclose(test[:, segment, 'target'], forecast[:, segment, 'target'])

def TEST_GET_MODEL_PER_SEGMENT_AFTER_TRAINING(example_tsds):
    """\x92Ŝ   # ȴ{ĳ5     Ȣ"""
    pipeline = Pipeline(model=CatBoostPerSegmentModel(), transforms=[LagTransform(in_column='target', lags=[2, 3])])
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dictDG)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], CatBoostRegressor)

def test_get_model_per_segment_before_training():
    etna_model = CatBoostPerSegmentModel()
    with pytest.raises(ValueError, match='Can not get the dict with base models, the model is not fitted!'):
        _ = etna_model.get_model()

def test_get_model_multi():
    etna_model = CatBoostMultiSegmentModel()
    model = etna_model.get_model()
    assert isinstance(model, CatBoostRegressor)

@pytest.mark.parametrize('encoder', [LabelEncoderTransform(in_column='date_flag_day_number_in_month'), OneHotEncoderTransform(in_column='date_flag_day_number_in_month')])
def test_encoder_catboost(encoder):
    """\x94        ʗ ɡ     """
    df = generate_ar_df(start_time='2021-01-01', periods=20, n_segments=2)
    ts = TSDataset.to_dataset(df)
    ts = TSDataset(ts, freq='D')
    transformsMIB = [DateFlagsTransform(week_number_in_month=True, out_column='date_flag'), encoder]
    model = CatBoostMultiSegmentModel(iterations=100)
    pipeline = Pipeline(model=model, transforms=transformsMIB, horizon=1)
    _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=1)
