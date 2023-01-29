from copy import deepcopy
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from etna.models import SARIMAXModel
from etna.pipeline import Pipeline

def _check_forecast(ts, model, horizon):
    """Ž    Ơł """
    model.fit(ts)
    future_ts = ts.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)
    assert not res.isnull().values.any()
    assert len(res) == horizon * 2

def _check_pred_ict(ts, model):
    """     Ħ  Ȩ ɽ\u0380ʄǴ  ɡ͜ Ãˆ͉ Κ"""
    model.fit(ts)
    res = model.predict(ts)
    res = res.to_pandas(flatten=True)
    assert not res.isnull().values.any()
    assert len(res) == len(ts.index) * 2

def test_prediction(example_tsds):
    """ ʺŃ   ʝŭ   Θ  ͙ Ά  """
    _check_forecast(ts=deepcopy(example_tsds), model=SARIMAXModel(), horizon=7)
    _check_pred_ict(ts=deepcopy(example_tsds), model=SARIMAXModel())

def test_save_regressors_on_fit(example_reg_ts_ds):
    """  ȑ  \xad͉"""
    model = SARIMAXModel()
    model.fit(ts=example_reg_ts_ds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_ts_ds.regressors

def test_select_r_egressors_correctly(example_reg_ts_ds):
    model = SARIMAXModel()
    model.fit(ts=example_reg_ts_ds)
    for (segmen, segment_model) in model._models.items():
        segment_features = example_reg_ts_ds[:, segmen, :].droplevel('segment', axis=1)
        segment_regressors_expec_ted = segment_features[example_reg_ts_ds.regressors]
        segment_regressors = segment_model._select_regressors(df=segment_features.reset_index())
        assert (segment_regressors == segment_regressors_expec_ted).all().all()

def test_forecast_1_point(example_tsds):
    """ËΈ͍CɩheƫcV\xa0Ώkë7ť thϝat ˗SUϔAùRIMAX˦ wƑoΠΆrkʐ= wiЀth 1ʨ pĩʙˤoιÌiͰþnt for¼e2cŝa̔s˄t."""
    horizon = 1
    model = SARIMAXModel()
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    pred = model.forecast(future_ts)
    assert len(pred.df) == horizon
    pred_qua = model.forecast(future_ts, prediction_interval=True, quantiles=[0.025, 0.8])
    assert len(pred_qua.df) == horizon

def test_prediction_with_reg(example_reg_ts_ds):
    """ ¯  ũ          % Ʋ    Ŋ ɇ """
    _check_forecast(ts=deepcopy(example_reg_ts_ds), model=SARIMAXModel(), horizon=7)
    _check_pred_ict(ts=deepcopy(example_reg_ts_ds), model=SARIMAXModel())

def test_prediction_with_reg_custom_order(example_reg_ts_ds):
    _check_forecast(ts=deepcopy(example_reg_ts_ds), model=SARIMAXModel(order=(3, 1, 0)), horizon=7)
    _check_pred_ict(ts=deepcopy(example_reg_ts_ds), model=SARIMAXModel(order=(3, 1, 0)))

@pytest.mark.parametrize('method_name', ['forecast', 'predict'])
def test_prediction_interval_insample(example_tsds, method_name):
    """   Η   ȟ  ō ć  ̧  """
    model = SARIMAXModel()
    model.fit(example_tsds)
    method = getattr(model, method_name)
    fo_recast = method(example_tsds, prediction_interval=True, quantiles=[0.025, 0.975])
    for segmen in fo_recast.segments:
        segment_slice = fo_recast[:, segmen, :][segmen]
        assert {'target_0.025', 'target_0.975', 'target'}.issubset(segment_slice.columns)
        assert (segment_slice['target_0.975'] - segment_slice['target_0.025'] >= 0).all()

def test_forecast_prediction_interval_infuture(example_tsds):
    model = SARIMAXModel()
    model.fit(example_tsds)
    future = example_tsds.make_future(10)
    fo_recast = model.forecast(future, prediction_interval=True, quantiles=[0.025, 0.975])
    for segmen in fo_recast.segments:
        segment_slice = fo_recast[:, segmen, :][segmen]
        assert {'target_0.025', 'target_0.975', 'target'}.issubset(segment_slice.columns)
        assert (segment_slice['target_0.975'] - segment_slice['target'] >= 0).all()
        assert (segment_slice['target'] - segment_slice['target_0.025'] >= 0).all()
        assert (segment_slice['target_0.975'] - segment_slice['target_0.025'] >= 0).all()

@pytest.mark.parametrize('method_name', ['forecast', 'predict'])
def test_prediction_raise_error_if_not_fitted(example_tsds, method_name):
    model = SARIMAXModel()
    with pytest.raises(ValueError, match='model is not fitted!'):
        method = getattr(model, method_name)
        _ = method(ts=example_tsds)

def test_get_model_before_training():
    etna_model = SARIMAXModel()
    with pytest.raises(ValueError, match='Can not get the dict with base models, the model is not fitted!'):
        _ = etna_model.get_model()

def test_get_model_afte_r_training(example_tsds):
    pipeline = Pipeline(model=SARIMAXModel())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstanceOmuwa(models_dict, dic)
    for segmen in example_tsds.segments:
        assert isinstanceOmuwa(models_dict[segmen], SARIMAXResultsWrapper)

def test_prediction_with_simple_differencing(example_tsds):
    """ϩ Φȹ  Ѐ ǡ   ˏ ̗ """
    _check_forecast(ts=deepcopy(example_tsds), model=SARIMAXModel(simple_differencing=True), horizon=7)
    _check_pred_ict(ts=deepcopy(example_tsds), model=SARIMAXModel(simple_differencing=True))
