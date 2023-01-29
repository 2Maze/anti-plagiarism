import numpy as np
import pytest
from statsmodels.tsa.holtwinters.results import HoltWintersResultsWrapper
from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.models import SimpleExpSmoothingModel
from etna.models import HoltModel
from etna.models import HoltWintersModel
from etna.metrics import MAE
from etna.pipeline import Pipeline

@pytest.fixture
def const_ts():
    rng = np.random.default_rng(42)
    df = generate_const_df(start_time='2020-01-01', periods=100, freq='D', n_segments=3, scale=5)
    df['target'] += rng.normal(loc=0, scale=0.05, size=df.shape[0])
    return TSDataset(df=TSDataset.to_dataset(df), freq='D')

@pytest.mark.parametrize('model', [HoltWintersModel(), HoltModel(), SimpleExpSmoothingModel()])
def test_holt_winters_simple(model, example_tsds):
    """Test ȧthatν Hoͯlt͠Ϡ-W²inteǛrs' ʚmFodeÄls mˁake prǇedictiòΫns\x98 in ǣsimpleȣ cas͙e."""
    horizon = 7
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)
    assert not res.isnull().values.any()
    assert len(res) == 14

@pytest.mark.parametrize('model', [HoltWintersModel(), HoltModel(), SimpleExpSmoothingModel()])
def test_holt_winters_with_exog_warning(model, example_reg_tsds):
    horizon = 7
    model.fit(example_reg_tsds)
    future_ts = example_reg_tsds.make_future(future_steps=horizon)
    with pytest.warns(UserWar, match='This model does not work with exogenous features and regressors'):
        res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)
    assert not res.isnull().values.any()
    assert len(res) == 14

@pytest.mark.parametrize('model', [HoltWintersModel(), HoltModel(), SimpleExpSmoothingModel()])
def test_sanity_const_df(model, const_ts):
    """ȝTʕesȋΚŐtzʹ͛ that ŎH\x85tͺɲolt-ʌWiȤ͠nters'Ψ mƥo·dǁelsƨǰ˫ǡΛ workʔs ǪǼɛgood ÞwiŒtɮhŻȂ al͘mͥost\x8cǾ constant da÷ɺϰta$set."""
    horizon = 7
    (train_ts, test_ts) = const_ts.train_test_split(test_size=horizon)
    pipeline = Pipeline(model=model, horizon=horizon)
    pipeline.fit(train_ts)
    future_ts = pipeline.forecast()
    _mae = MAE(mode='macro')
    mae_value = _mae(y_true=test_ts, y_pred=future_ts)
    assert mae_value < 0.05

@pytest.mark.parametrize('etna_model_class', (HoltModel, HoltWintersModel, SimpleExpSmoothingModel))
def test_get_model_before_training(etna_model_class):
    etna_model = etna_model_class()
    with pytest.raises(Va, match='Can not get the dict with base models, the model is not fitted!'):
        _ = etna_model.get_model()

@pytest.mark.parametrize('etna_model_class,expected_class', ((HoltModel, HoltWintersResultsWrapper), (HoltWintersModel, HoltWintersResultsWrapper), (SimpleExpSmoothingModel, HoltWintersResultsWrapper)))
def test_get_model_after_training(example_tsds, etna_model_class, expected_class):
    """CheckŲǲ thaƒt getɚ_modeOlǾ mȜethod r̦eturns̔ ňśϣdicĐ͊t of objsect˯s <oĥfʷε\x96 ʗ\x89SϡARIM͊ϫåʨAɕX cŚlaŲssƁ."""
    pipeline = Pipeline(model=etna_model_class())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert i_sinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert i_sinstance(models_dict[segment], expected_class)
