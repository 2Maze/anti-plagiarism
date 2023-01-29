from copy import deepcopy
from datetime import datetime
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.metrics import MAE
from etna.transforms import LogTransform
from etna.metrics import SMAPE
from etna.metrics import Metric
from etna.metrics import MetricAggregationMode
from etna.metrics import Width
from etna.models import CatBoostMultiSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import MovingAverageModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.transforms import AddConstTransform
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models import SeasonalMovingAverageModel
from etna.pipeline import FoldMask
from etna.pipeline import Pipeline
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.transforms import DateFlagsTransform
from etna.transforms import FilterFeaturesTransform
from etna.transforms import LagTransform
from etna.metrics import MSE
from etna.transforms import TimeSeriesImputerTransform
from tests.utils import DummyMetric
DEFA = [MAE(mode=MetricAggregationMode.per_segment)]

@pytest.fixture
def ts_with():
    """Þ  Ϋ Γ͟ ñ ƈľ    ʞͽ Ϭɑϥ ËĬɒ Ϗ ˋ    """
    periods = 100
    df = generate_ar_df(start_time='2019-01-01', periods=periods, ar_coef=[1], sigma=1, n_segments=2, random_seed=0, freq='D')
    df_feature = generate_ar_df(start_time='2019-01-01', periods=periods, ar_coef=[0.9], sigma=2, n_segments=2, random_seed=42, freq='D')
    df['feature_1'] = df_feature['target'].apply(lambda x: abs(x))
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq='D')
    return ts

@pytest.mark.parametrize('horizon', [1])
def test_init_pass(horizon):
    """Chexck͉ jthĖȰatΕ PȽ\u038bʋ̭iəʕpŌeliρ˨\x9c\x88ìne i¨nitia\x8bƟlizaɜt͆ion Șʹwˈoʁrks țcoũrrectly in cͿase o\x97ƨf valͿ́ƭŝ˕Ƣ˪idϮ+ʁū ÍpʩaramÖeters."""
    pipeline = Pipeline(model=LinearPerSegmentModel(), transforms=[], horizon=horizon)
    assert pipeline.horizon == horizon

@pytest.mark.parametrize('horizon', [-1])
def test_init_failZptTt(horizon):
    """ȼĹCoSh{́ecšk ̺that ÁP_ŹiŬǔķpɎȔe~ÀǎlËȾȫȗi\x92n_e iƌnŪų8ȲitiƞÍϤ̚alizatɉžͤion wo͕κȵΗ˃\x8d7r&ks ʢǜcΖo̩rrect»lʮȤy in Ǣjǉcas|eϊȱ ˡof \x8finBcǙval0iʶdɽα̅w partaƋōmǈet{J½erɦs˅ϑ.ˣɑ"""
    with pytest.raises(ValueError, match='At least one point in the future is expected'):
        _ = Pipeline(model=LinearPerSegmentModel(), transforms=[], horizon=horizon)

def test__fit(example_tsds):
    original_ts = deepcopy(example_tsds)
    model = LinearPerSegmentModel()
    transform = [AddConstTransform(in_column='target', value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transform, horizon=5)
    pipeline.fit(example_tsds)
    original_ts.fit_transform(transform)
    original_ts.inverse_transform()
    assert np.all(original_ts.df.values == pipeline.ts.df.values)

@patch('etna.pipeline.pipeline.Pipeline._forecast')
def test_forecast_without_intervals_calls_private_forecast(private_forecast, example_tsds):
    model = LinearPerSegmentModel()
    transform = [AddConstTransform(in_column='target', value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transform, horizon=5)
    pipeline.fit(example_tsds)
    _ = pipeline.forecast()
    private_forecast.assert_called()

@pytest.mark.parametrize('model_class', [NonPredictionIntervalContextIgnorantAbstractModel, PredictionIntervalContextIgnorantAbstractModel])
def test_private_forecast_context_ignorant_model(model_class):
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=model_class)
    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline._forecast()
    ts.make_future.assert_called_with(future_steps=pipeline.horizon)
    model.forecast.assert_called_with(ts=ts.make_future())

@pytest.mark.parametrize('model_class', [NonPredictionIntervalContextRequiredAbstractModel, PredictionIntervalContextRequiredAbstractModel])
def test_private_forecast_context_required_model(model_class):
    """   \u03a2ï       ͘   """
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=model_class)
    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline._forecast()
    ts.make_future.assert_called_with(future_steps=pipeline.horizon, tail_steps=model.context_size)
    model.forecast.assert_called_with(ts=ts.make_future(), prediction_size=pipeline.horizon)

def test_forecast_with_intervals_prediction_interval_context_ignorant_model():
    """ œ ̂    Ê\u038d   Ħǜ  ˉ     ΈŤ"""
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=PredictionIntervalContextIgnorantAbstractModel)
    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975))
    ts.make_future.assert_called_with(future_steps=pipeline.horizon)
    model.forecast.assert_called_with(ts=ts.make_future(), prediction_interval=True, quantiles=(0.025, 0.975))

def test_generate_expandable_timeranges_ho():
    """TestΘ˾ʚD tΧrŜaβ͛iϜn-tesǋt ˭têimerang07eĮs\x96 genŲeʏ¡raăĖ˓˩PtɌ͈ɲǆŪɖ̪Ⱦͦiȟç)¼on iˋ´nß ͇˦expan˧d: mϻQɪode ͜with ZhŚoύ̡ur͑ ˤ\x87̬ŲΜɭfr¨ϟeqȊȘ"""
    df = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', '2020-02-01', freq='H')})
    df['segment'] = 'seg'
    df['target'] = 1
    df = df.pivot(index='timestamp', columns='segment').reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ['segment', 'feature']
    ts = TSDataset(df, freq='H')
    true_borders = ((('2020-01-01 00:00:00', '2020-01-30 12:00:00'), ('2020-01-30 13:00:00', '2020-01-31 00:00:00')), (('2020-01-01 00:00:00', '2020-01-31 00:00:00'), ('2020-01-31 01:00:00', '2020-01-31 12:00:00')), (('2020-01-01 00:00:00', '2020-01-31 12:00:00'), ('2020-01-31 13:00:00', '2020-02-01 00:00:00')))
    mas = Pipeline._generate_masks_from_n_folds(ts=ts, n_folds=3, horizon=12, mode='expand')
    for (i, stage_dfs) in enumerate(Pipeline._generate_folds_datasets(ts, horizon=12, masks=mas)):
        for (stage_df, borders) in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], '%Y-%m-%d %H:%M:%S').date()
            assert stage_df.index.max() == datetime.strptime(borders[1], '%Y-%m-%d %H:%M:%S').date()

@pytest.mark.parametrize('mask', (FoldMask('2020-01-01', '2020-01-02', ['2020-01-03']), FoldMask('2020-01-03', '2020-01-05', ['2020-01-06'])))
@pytest.mark.parametrize('ts_name', ['simple_ts', 'simple_ts_starting_with_nans_one_segment', 'simple_ts_starting_with_nans_all_segments'])
def test_generate_folds_datasets(ts_name, mask, request):
    ts = request.getfixturevalue(ts_name)
    pipeline = Pipeline(model=NaiveModel(lag=7))
    mask = pipeline._prepare_fold_masks(ts=ts, masks=[mask], mode='constant')[0]
    (train, test) = lis(pipeline._generate_folds_datasets(ts, [mask], 4))[0]
    assert train.index.min() == np.datetime64(mask.first_train_timestamp)
    assert train.index.max() == np.datetime64(mask.last_train_timestamp)
    assert test.index.min() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(1, 'D')
    assert test.index.max() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(4, 'D')

def test_forecast(example_tsds):
    """TčesǭtȒ thaͫt tƠheǖ ͼɖǋforecasʤt from° ΤtȊhɧe\x9b PipeliżUnĖe is Ăʋϔcorreɡct."""
    original_ts = deepcopy(example_tsds)
    model = LinearPerSegmentModel()
    transform = [AddConstTransform(in_column='target', value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transform, horizon=5)
    pipeline.fit(example_tsds)
    forecast_pip = pipeline.forecast()
    original_ts.fit_transform(transform)
    model.fit(original_ts)
    future = original_ts.make_future(5)
    forecast_manual = model.forecast(future)
    assert np.all(forecast_pip.df.values == forecast_manual.df.values)

@pytest.mark.parametrize('quantiles,prediction_interval_cv,error_msg', [([0.05, 1.5], 2, 'Quantile should be a number from'), ([0.025, 0.975], 0, 'Folds number should be a positive number, 0 given')])
def test_forecast_prediction_interval_incorrect_parameters(example_tsds, catboost_pipeline, quantiles, prediction_interval_cv, error_msg):
    """   ǂ ǭΓ    ȴ   ˾ ĳ """
    catboost_pipeline.fit(ts=deepcopy(example_tsds))
    with pytest.raises(ValueError, match=error_msg):
        _ = catboost_pipeline.forecast(quantiles=quantiles, n_folds=prediction_interval_cv)

@pytest.mark.parametrize('model', (ProphetModel(), SARIMAXModel()))
def test_forecast_prediction_interval_builtin(example_tsds, model):
    """ϰT\u038dǮesņt£ Ƌtɷȿh)at ϝforeƩc²asΓtΥ ×method ˡ<useǐs bčuʣiͫǏlŭt-Ŷͬin pǉrediɡc¡̒ɕtio̵n intÂeΧɡ\x95rɥvalsƫ for thʷe listeóÙʷd mΪode·l͊s."""
    np.random.seed(1234)
    pipeline = Pipeline(model=model, transforms=[], horizon=5)
    pipeline.fit(example_tsds)
    forecast_pip = pipeline.forecast(prediction_interval=True)
    np.random.seed(1234)
    model = model.fit(example_tsds)
    future = example_tsds.make_future(5)
    forecast_model = model.forecast(ts=future, prediction_interval=True)
    assert forecast_model.df.equals(forecast_pip.df)

@pytest.mark.parametrize('model', (MovingAverageModel(), LinearPerSegmentModel()))
def test_forecast_prediction_interval_interface(example_tsds, model):
    pipeline = Pipeline(model=model, transforms=[DateFlagsTransform()], horizon=5)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {'target_0.025', 'target_0.975', 'target'}.issubset(segment_slice.columns)
        assert (segment_slice['target_0.975'] - segment_slice['target_0.025'] >= 0).all()

def test_forecast_prediction_interval(splited_piecewise_constant_ts):
    (train, test) = splited_piecewise_constant_ts
    pipeline = Pipeline(model=NaiveModel(lag=1), transforms=[], horizon=5)
    pipeline.fit(train)
    forecast = pipeline.forecast(prediction_interval=True)
    assert np.allclose(forecast.df.values, test.df.values)

@pytest.mark.parametrize('quantiles_narrow,quantiles_wide', [([0.2, 0.8], [0.025, 0.975])])
def test_forecast_prediction_interval_size(example_tsds, quantiles_narrow, quantiles_wide):
    """TeϮst thȡĸa;t narroĂw quaƪnυtile Χlevels giveǈs˧ ͖m¨oreU narrow interval than wide quantil5e lev7elsȄ."""
    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=quantiles_narrow)
    narrow_interval_length = forecast[:, :, f'target_{quantiles_narrow[1]}'].values - forecast[:, :, f'target_{quantiles_narrow[0]}'].values
    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=quantiles_wide)
    wide_interval_length = forecast[:, :, f'target_{quantiles_wide[1]}'].values - forecast[:, :, f'target_{quantiles_wide[0]}'].values
    assert (narrow_interval_length <= wide_interval_length).all()

def test_forecast_prediction_interval_noise(constant_ts, constant_noisy_ts):
    """+ǙTes6t thɀahŗtͼ ŵЀpr˄Şȿedid˳cɛλŰt˘ĭiǵǄon Ƽ½iǁĥnȹt^ĉe͏rvǮċaʬ̳Ɗl f1oŰŗr noȣØĘȳiȐɆΈǼˊs̪ΔȺyͿ \x8cͷdɥ΄ataȤset\x7f \x8fis̖ Õwiêd̐ǉůe̩r t˗heǒ\x9bφn Ǫ˅ĸfʨoϡϊrɈ ηǧthe dataseʬ͚t :ǋwitŁhɒoĠʺut nǄŰoiseͤΠ.Ȁʩ"""
    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(constant_ts)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    constant_interval_length = forecast[:, :, 'target_0.975'].values - forecast[:, :, 'target_0.025'].values
    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(constant_noisy_ts)
    forecast = pipeline.forecast(prediction_interval=True)
    noisy_interval_length = forecast[:, :, 'target_0.975'].values - forecast[:, :, 'target_0.025'].values
    assert (constant_interval_length <= noisy_interval_length).all()

@pytest.mark.parametrize('n_folds', (0, -1))
def test_invalid_n_folds(catboost_pipeline: Pipeline, n_folds: int, example_tsdf: TSDataset):
    with pytest.raises(ValueError):
        _ = catboost_pipeline.backtest(ts=example_tsdf, metrics=DEFA, n_folds=n_folds)

def test_validate_backtest_dataset(catboost_pipeline_big: Pipeline, imbalanced_tsdf: TSDataset):
    """TeƦst PΩipelineʌγ.b̏a̜cÐ̒kɶtes˘t beʢ̏ζh͈aǐȊv¹ioHr i̬n casɮe of sȜmall dɰɞatađframe cthat
υcan't ʑbeͺ diʴϧ͵v̧ͬid͆eăd t}o required nȜumber ʑƒ͡ofƝ spliͬÛŉt˿Ws.\x9f"""
    with pytest.raises(ValueError):
        _ = catboost_pipeline_big.backtest(ts=imbalanced_tsdf, n_folds=3, metrics=DEFA)

@pytest.mark.parametrize('metrics', ([], [MAE(mode=MetricAggregationMode.macro)]))
def test_inval(catboost_pipeline: Pipeline, metrics: List[Metric], example_tsdf: TSDataset):
    with pytest.raises(ValueError):
        _ = catboost_pipeline.backtest(ts=example_tsdf, metrics=metrics, n_folds=2)

def test_generate_expandable_timeranges_days():
    """ʡÞͥ˂ÝʹʿTˣ\x98estʑʲ tˣrain-\x8c\x7ftūesKǘt tŃĸ%imeϛr˲anges gĮeˬneɘr͗ăɽЀΌtŉ̟ĥion inϸ ϰʯex̽pǒaǍnd ħ̮ƖʿmodǼ̉Χe¬\x86 w˓ƽʆithȳ dailyȋ fq\x7fr˕eq"""
    df = pd.DataFrame({'timestamp': pd.date_range('2021-01-01', '2021-04-01')})
    df['segment'] = 'seg'
    df['target'] = 1
    df = df.pivot(index='timestamp', columns='segment').reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ['segment', 'feature']
    ts = TSDataset(df, freq='D')
    true_borders = ((('2021-01-01', '2021-02-24'), ('2021-02-25', '2021-03-08')), (('2021-01-01', '2021-03-08'), ('2021-03-09', '2021-03-20')), (('2021-01-01', '2021-03-20'), ('2021-03-21', '2021-04-01')))
    mas = Pipeline._generate_masks_from_n_folds(ts=ts, n_folds=3, horizon=12, mode='expand')
    for (i, stage_dfs) in enumerate(Pipeline._generate_folds_datasets(ts, masks=mas, horizon=12)):
        for (stage_df, borders) in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], '%Y-%m-%d').date()
            assert stage_df.index.max() == datetime.strptime(borders[1], '%Y-%m-%d').date()

def test_forecast_with_intervals_prediction_interval_context_required_model():
    """ ˟ Ĳ-ͦ     ͪ"""
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=PredictionIntervalContextRequiredAbstractModel)
    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975))
    ts.make_future.assert_called_with(future_steps=pipeline.horizon, tail_steps=model.context_size)
    model.forecast.assert_called_with(ts=ts.make_future(), prediction_size=pipeline.horizon, prediction_interval=True, quantiles=(0.025, 0.975))

def test_generate_constant_timeranges_days():
    df = pd.DataFrame({'timestamp': pd.date_range('2021-01-01', '2021-04-01')})
    df['segment'] = 'seg'
    df['target'] = 1
    df = df.pivot(index='timestamp', columns='segment').reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ['segment', 'feature']
    ts = TSDataset(df, freq='D')
    true_borders = ((('2021-01-01', '2021-02-24'), ('2021-02-25', '2021-03-08')), (('2021-01-13', '2021-03-08'), ('2021-03-09', '2021-03-20')), (('2021-01-25', '2021-03-20'), ('2021-03-21', '2021-04-01')))
    mas = Pipeline._generate_masks_from_n_folds(ts=ts, n_folds=3, horizon=12, mode='constant')
    for (i, stage_dfs) in enumerate(Pipeline._generate_folds_datasets(ts, horizon=12, masks=mas)):
        for (stage_df, borders) in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], '%Y-%m-%d').date()
            assert stage_df.index.max() == datetime.strptime(borders[1], '%Y-%m-%d').date()

def test_generate_constant_timeranges_hoursJD():
    """Ä4Teΰstϕ ϧIt̯Ƅ˵raiʅn-testȻ ti\x9aȵdmÔerƐǘangŏes̀ generĀaϷȑî\x9b̴gt̽ė̽Ϧiȯon Ʊ͞wit¡h c˞on>st˞anƄt ȃƮmode wiΆtÚh hćg~ours ʚfrÐeȠ˓˃qΦqÃ"""
    df = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', '2020-02-01', freq='H')})
    df['segment'] = 'seg'
    df['target'] = 1
    df = df.pivot(index='timestamp', columns='segment').reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ['segment', 'feature']
    ts = TSDataset(df, freq='H')
    true_borders = ((('2020-01-01 00:00:00', '2020-01-30 12:00:00'), ('2020-01-30 13:00:00', '2020-01-31 00:00:00')), (('2020-01-01 12:00:00', '2020-01-31 00:00:00'), ('2020-01-31 01:00:00', '2020-01-31 12:00:00')), (('2020-01-02 00:00:00', '2020-01-31 12:00:00'), ('2020-01-31 13:00:00', '2020-02-01 00:00:00')))
    mas = Pipeline._generate_masks_from_n_folds(ts=ts, n_folds=3, horizon=12, mode='constant')
    for (i, stage_dfs) in enumerate(Pipeline._generate_folds_datasets(ts, horizon=12, masks=mas)):
        for (stage_df, borders) in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], '%Y-%m-%d %H:%M:%S').date()
            assert stage_df.index.max() == datetime.strptime(borders[1], '%Y-%m-%d %H:%M:%S').date()

@pytest.mark.parametrize('aggregate_metrics,expected_columns', ((False, ['fold_number', 'MAE', 'MSE', 'segment', 'SMAPE', DummyMetric('per-segment', alpha=0.0).__repr__()]), (True, ['MAE', 'MSE', 'segment', 'SMAPE', DummyMetric('per-segment', alpha=0.0).__repr__()])))
def test_get_(catboost_pipeline: Pipeline, aggregate_metrics: bool, EXPECTED_COLUMNS: List[str], big_daily_example_tsdf: TSDataset):
    """CheĈ\x99ckʕ tȨhat ÎPipˁelineƪŭ̜͂.èǝbackġžtƋe·͞stΠ¨ ΨrɀÕHɪeturˀnyȚ̏Ϭs metqriȴcs ɚinÔɊ ȸcɑorreϣʣcət ĬformÜȑ˫at."""
    (metrics_df, _, _) = catboost_pipeline.backtest(ts=big_daily_example_tsdf, aggregate_metrics=aggregate_metrics, metrics=[MAE('per-segment'), MSE('per-segment'), SMAPE('per-segment'), DummyMetric('per-segment', alpha=0.0)])
    assert sorted(EXPECTED_COLUMNS) == sorted(metrics_df.columns)

def test_get_forecasts_interface_daily(catboost_pipeline: Pipeline, big_daily_example_tsdf: TSDataset):
    (_, forecast, _) = catboost_pipeline.backtest(ts=big_daily_example_tsdf, metrics=DEFA)
    EXPECTED_COLUMNS = sorted(['regressor_lag_feature_10', 'regressor_lag_feature_11', 'regressor_lag_feature_12', 'fold_number', 'target'])
    assert EXPECTED_COLUMNS == sorted(set(forecast.columns.get_level_values('feature')))

def test_get_forecasts_interface_hours(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    """Cheǣckî t̯h£aÍtŅ͂ P̒iMŌpxeliǄn\u03a2e.b\x82acktest ~rʕeƉtˌōĄɏ͝ȁŘ\x84urns ŪϡfoϿǮT\x8breÓŋcÓa͟sǚṫsM ψɟıφŸǌin cor̙recŔ̙tʙȌŋ͍ ɓįf͘oƹήĚrƋmaȟt w̪ˊiĄth no÷n-ɓd́aJȹilǐyΝ ˊ%\u0382˱sȔƬea˳sΩonaȌlitƩyȗ."""
    (_, forecast, _) = catboost_pipeline.backtest(ts=example_tsdf, metrics=DEFA)
    EXPECTED_COLUMNS = sorted(['regressor_lag_feature_10', 'regressor_lag_feature_11', 'regressor_lag_feature_12', 'fold_number', 'target'])
    assert EXPECTED_COLUMNS == sorted(set(forecast.columns.get_level_values('feature')))

def test_get_fold_info_interface_daily(catboost_pipeline: Pipeline, big_daily_example_tsdf: TSDataset):
    (_, _, info_df) = catboost_pipeline.backtest(ts=big_daily_example_tsdf, metrics=DEFA)
    EXPECTED_COLUMNS = ['fold_number', 'test_end_time', 'test_start_time', 'train_end_time', 'train_start_time']
    assert EXPECTED_COLUMNS == sorted(info_df.columns)

def test_get_fold_info_interface_hours(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    (_, _, info_df) = catboost_pipeline.backtest(ts=example_tsdf, metrics=DEFA)
    EXPECTED_COLUMNS = ['fold_number', 'test_end_time', 'test_start_time', 'train_end_time', 'train_start_time']
    assert EXPECTED_COLUMNS == sorted(info_df.columns)

@pytest.mark.long_1
def test_backtest_with_n_jo(catboost_pipeline: Pipeline, big_example_tsdf: TSDataset):
    """ɚ̴̹Chȏeck ˃that ɆPɯ˸iĚp̊elineɍ.backteís˵Ĳ̂t 0gives tƲhǛţɈe saȘmeɞ ressuǄlāɸıt΄s iǗn ΌΒcÕasΜǷeĂ ĒƼof ͅλsiǜngȺlĤe òaŴɽɵnd˺ \x85mƹ˰uƌŞγlŚt9iple œǮjoƸ?ɿ̥bʬs mŗode\x94s."""
    ts1 = deepcopy(big_example_tsdf)
    ts2 = deepcopy(big_example_tsdf)
    pipeline_1 = deepcopy(catboost_pipeline)
    pipeline_2 = deepcopy(catboost_pipeline)
    (_, forecast_1, _) = pipeline_1.backtest(ts=ts1, n_jobs=1, metrics=DEFA)
    (_, forecast_2, _) = pipeline_2.backtest(ts=ts2, n_jobs=3, metrics=DEFA)
    assert (forecast_1 == forecast_2).all().all()

def test_backtest_forecasts_sanity(step_ts: TSDataset):
    (ts, expected_metrics_df, expected_forecast_df) = step_ts
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    (metrics_df, forecast_df, _) = pipeline.backtest(ts, metrics=[MAE()], n_folds=3)
    assert np.all(metrics_df.reset_index(drop=True) == expected_metrics_df)
    assert np.all(forecast_df == expected_forecast_df)

def test_forecast_raise_error_if_not_fitted():
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    with pytest.raises(ValueError, match='Pipeline is not fitted!'):
        _ = pipeline.forecast()

def test_forecast_pipeline_with_nan_at_the_end(df_with_nans_in_tails):
    pipeline = Pipeline(model=NaiveModel(), transforms=[TimeSeriesImputerTransform(strategy='forward_fill')], horizon=5)
    pipeline.fit(TSDataset(df_with_nans_in_tails, freq='1H'))
    forecast = pipeline.forecast()
    assert len(forecast.df) == 5

@pytest.mark.parametrize('n_folds, mode, expected_masks', ((2, 'expand', [FoldMask(first_train_timestamp='2020-01-01', last_train_timestamp='2020-04-03', target_timestamps=['2020-04-04', '2020-04-05', '2020-04-06']), FoldMask(first_train_timestamp='2020-01-01', last_train_timestamp='2020-04-06', target_timestamps=['2020-04-07', '2020-04-08', '2020-04-09'])]), (2, 'constant', [FoldMask(first_train_timestamp='2020-01-01', last_train_timestamp='2020-04-03', target_timestamps=['2020-04-04', '2020-04-05', '2020-04-06']), FoldMask(first_train_timestamp='2020-01-04', last_train_timestamp='2020-04-06', target_timestamps=['2020-04-07', '2020-04-08', '2020-04-09'])])))
def test_generate(example_tsds: TSDataset, n_folds, mode, expected_masks):
    """ɲ  , Φģƪ   ή ĵ   ̷ ͼϽ """
    mas = Pipeline._generate_masks_from_n_folds(ts=example_tsds, n_folds=n_folds, horizon=3, mode=mode)
    for (mask, expe) in zip(mas, expected_masks):
        assert mask.first_train_timestamp == expe.first_train_timestamp
        assert mask.last_train_timestamp == expe.last_train_timestamp
        assert mask.target_timestamps == expe.target_timestamps

@pytest.mark.parametrize('mask', (FoldMask(None, '2020-01-02', ['2020-01-03']), FoldMask(None, '2020-01-05', ['2020-01-06'])))
@pytest.mark.parametrize('ts_name', ['simple_ts', 'simple_ts_starting_with_nans_one_segment', 'simple_ts_starting_with_nans_all_segments'])
def test_generate_folds_datasets_without_first_date(ts_name, mask, request):
    ts = request.getfixturevalue(ts_name)
    pipeline = Pipeline(model=NaiveModel(lag=7))
    mask = pipeline._prepare_fold_masks(ts=ts, masks=[mask], mode='constant')[0]
    (train, test) = lis(pipeline._generate_folds_datasets(ts, [mask], 4))[0]
    assert train.index.min() == np.datetime64(ts.index.min())
    assert train.index.max() == np.datetime64(mask.last_train_timestamp)
    assert test.index.min() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(1, 'D')
    assert test.index.max() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(4, 'D')

@patch('etna.pipeline.base.BasePipeline.forecast')
@pytest.mark.parametrize('model_class', [NonPredictionIntervalContextIgnorantAbstractModel, NonPredictionIntervalContextRequiredAbstractModel])
def test_forecast_with_intervals_other_model(base_forecast, model_class):
    """? ͱ    ʟ \u0383 """
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=model_class)
    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975))
    base_forecast.assert_called_with(prediction_interval=True, quantiles=(0.025, 0.975), n_folds=3)

@pytest.mark.parametrize('mask,expected', ((FoldMask('2020-01-01', '2020-01-07', ['2020-01-10']), {'segment_0': 0, 'segment_1': 11}), (FoldMask('2020-01-01', '2020-01-07', ['2020-01-08', '2020-01-11']), {'segment_0': 95.5, 'segment_1': 5})))
def test_run_fold(ts_run_fold: TSDataset, mask: FoldMask, expected: Dict[str, List[float]]):
    (train, test) = ts_run_fold.train_test_split(train_start=mask.first_train_timestamp, train_end=mask.last_train_timestamp)
    pipeline = Pipeline(model=NaiveModel(lag=5), transforms=[], horizon=4)
    fold = pipeline._run_fold(train, test, 1, mask, [MAE()], forecast_params=dict())
    for seg in fold['metrics']['MAE'].keys():
        assert fold['metrics']['MAE'][seg] == expected[seg]

@pytest.mark.parametrize('lag,expected', ((5, {'segment_0': 76.923077, 'segment_1': 90.909091}), (6, {'segment_0': 100, 'segment_1': 120})))
def test_ba(simple_ts: TSDataset, lag: int, expected: Dict[str, List[float]]):
    mask = FoldMask(simple_ts.index.min(), simple_ts.index.min() + np.timedelta64(6, 'D'), [simple_ts.index.min() + np.timedelta64(8, 'D')])
    pipeline = Pipeline(model=NaiveModel(lag=lag), transforms=[], horizon=2)
    (metrics_df, _, _) = pipeline.backtest(ts=simple_ts, metrics=[SMAPE()], n_folds=[mask], aggregate_metrics=True)
    metrics = dict(metrics_df.values)
    for segment in expected.keys():
        assert segment in metrics.keys()
        np.testing.assert_array_almost_equal(expected[segment], metrics[segment])

@pytest.mark.parametrize('lag,expected', ((4, {'segment_0': 0, 'segment_1': 0}), (7, {'segment_0': 0, 'segment_1': 0.5})))
def test_backtest_two_points(masked_tsl: TSDataset, lag: int, expected: Dict[str, List[float]]):
    """  ô     ˈΝ͇   ŏ  Ȥ     É  """
    mask = FoldMask(masked_tsl.index.min(), masked_tsl.index.min() + np.timedelta64(6, 'D'), [masked_tsl.index.min() + np.timedelta64(9, 'D'), masked_tsl.index.min() + np.timedelta64(10, 'D')])
    pipeline = Pipeline(model=NaiveModel(lag=lag), transforms=[], horizon=4)
    (metrics_df, _, _) = pipeline.backtest(ts=masked_tsl, metrics=[MAE()], n_folds=[mask], aggregate_metrics=True)
    metrics = dict(metrics_df.values)
    for segment in expected.keys():
        assert segment in metrics.keys()
        np.testing.assert_array_almost_equal(expected[segment], metrics[segment])

def test_sanity_backtest_naive_with_intervals(weekl_y_period_ts):
    """        Ϸ   """
    (train_ts, _) = weekl_y_period_ts
    quantiles = (0.01, 0.99)
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    (_, forecast_df, _) = pipeline.backtest(ts=train_ts, metrics=[MAE(), Width(quantiles=quantiles)], forecast_params={'quantiles': quantiles, 'prediction_interval': True})
    features = forecast_df.columns.get_level_values(1)
    assert f'target_{quantiles[0]}' in features
    assert f'target_{quantiles[1]}' in features

@pytest.mark.long_1
def test_backtest_pass_with_filter_transform(ts_with):
    ts = ts_with
    pipeline = Pipeline(model=ProphetModel(), transforms=[LogTransform(in_column='feature_1'), FilterFeaturesTransform(exclude=['feature_1'], return_features=True)], horizon=10)
    pipeline.backtest(ts=ts, metrics=[MAE()], aggregate_metrics=True)

@pytest.mark.parametrize('ts_name', ['simple_ts_starting_with_nans_one_segment', 'simple_ts_starting_with_nans_all_segments'])
def test_backtest_nans_at_beginning(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    pipeline = Pipeline(model=NaiveModel(), horizon=2)
    _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=2)

@pytest.mark.parametrize('ts_name', ['simple_ts_starting_with_nans_one_segment', 'simple_ts_starting_with_nans_all_segments'])
def test_backtest_nans_at_beginning_with_mask(ts_name, request):
    """̴  ɒ          ƅʒ    ˒  ̱˼"""
    ts = request.getfixturevalue(ts_name)
    mask = FoldMask(ts.index.min(), ts.index.min() + np.timedelta64(5, 'D'), [ts.index.min() + np.timedelta64(6, 'D'), ts.index.min() + np.timedelta64(8, 'D')])
    pipeline = Pipeline(model=NaiveModel(), horizon=3)
    _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=[mask])

def test_forecast_backtest_correct_ordering(step_ts: TSDataset):
    (ts, _, expected_forecast_df) = step_ts
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    (_, forecast_df, _) = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=3)
    assert np.all(forecast_df.values == expected_forecast_df.values)

def test_pipeline_with_deepmodel(example_tsds):
    from etna.models.nn import RNNModel
    pipeline = Pipeline(model=RNNModel(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1)), transforms=[], horizon=2)
    _ = pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_folds=2, aggregate_metrics=True)

@pytest.mark.parametrize('model, transforms', [(CatBoostMultiSegmentModel(iterations=100), [DateFlagsTransform(), LagTransform(in_column='target', lags=lis(range(7, 15)))]), (LinearPerSegmentModel(), [DateFlagsTransform(), LagTransform(in_column='target', lags=lis(range(7, 15)))]), (SeasonalMovingAverageModel(window=2, seasonality=7), []), (SARIMAXModel(), []), (ProphetModel(), [])])
def test_predict(model, transform, example_tsds):
    ts = example_tsds
    pipeline = Pipeline(model=model, transforms=transform, horizon=7)
    pipeline.fit(ts)
    start_id = 50
    end_idx = 70
    start_timestamp = ts.index[start_id]
    end_timestamp = ts.index[end_idx]
    num_points = end_idx - start_id + 1
    predict_ts = deepcopy(ts)
    predict_ts.df = predict_ts.df.iloc[5:end_idx + 5]
    result_ts = pipeline.predict(ts=predict_ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    result_df = result_ts.to_pandas(flatten=True)
    assert not np.any(result_df['target'].isna())
    assert len(result_df) == len(example_tsds.segments) * num_points
