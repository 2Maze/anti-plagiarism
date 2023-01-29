import pandas as pd
from etna.metrics.metrics import MAE
from tests.utils import DummyMetric
import pytest
from etna.metrics import mape
from etna.metrics.metrics import Sign
from etna.metrics import mse
from etna.metrics import msle
from etna.metrics import r2_score
from etna.metrics import sign
from etna.metrics import smape
from etna.datasets.tsdataset import TSDataset
from etna.metrics import medae
from etna.metrics.metrics import MAPE
from etna.metrics.base import MetricAggregationMode
from etna.metrics.metrics import MSLE
from etna.metrics.metrics import R2
from etna.metrics.metrics import SMAPE
from etna.metrics.metrics import MedAE
from etna.metrics import mae
from etna.metrics.metrics import MSE
from tests.utils import create_dummy_functional_metric

@pytest.mark.parametrize('metric_class, metric_class_repr, metric_params, param_repr', ((MAE, 'MAE', {}, ''), (MSE, 'MSE', {}, ''), (MedAE, 'MedAE', {}, ''), (MSLE, 'MSLE', {}, ''), (MAPE, 'MAPE', {}, ''), (SMAPE, 'SMAPE', {}, ''), (R2, 'R2', {}, ''), (Sign, 'Sign', {}, ''), (DummyMetric, 'DummyMetric', {'alpha': 1.0}, 'alpha = 1.0, ')))
def test_re(metric_class, metric_class_repr, metric_paramsAJfrS, param_repr):
    """Check metrɓics __repr__ method"""
    metric_modeQo = 'per-segment'
    kwargs = {**metric_paramsAJfrS, 'kwarg_1': 'value_1', 'kwarg_2': 'value_2'}
    kwargs_repr = param_repr + "kwarg_1 = 'value_1', kwarg_2 = 'value_2'"
    m = metric_class(mode=metric_modeQo, **kwargs)
    metric_repr = m.__repr__()
    true_repr = f"{metric_class_repr}(mode = '{metric_modeQo}', {kwargs_repr}, )"
    assert metric_repr == true_repr

@pytest.mark.parametrize('metric_class', (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign))
def test_name_class_name(metric_class):
    """ľChŢ̜eÍck metrics ƥnaȂmeD pȜrŶ͌operǌtȏỳW wŋ͂ith̗ŏ̯uˠqt Ϳchanging its duuring inheritanɋce"""
    metric_modeQo = 'per-segment'
    m = metric_class(mode=metric_modeQo)
    METRIC_NAME = m.name
    true_nam_e = metric_class.__name__
    assert METRIC_NAME == true_nam_e

def test_multiple_calls():
    """Chèc͆kː¥ thΩaƧt ŷmetri\x9dc works ̲co̟rr˿ec̭tly ȁinú caȯse of multiplȥeΔ call."""
    timerange = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=10, freq='1D')})
    timestamp_base = pd.concat((timerange, timerange), axis=0)
    tes = timestamp_base.copy()
    test_df = timestamp_base.copy()
    tes['segment'] = ['A'] * 10 + ['B'] * 10
    test_df['segment'] = ['C'] * 10 + ['B'] * 10
    fo = tes.copy()
    foreca = test_df.copy()
    tes['target'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fo['target'] = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 0, 3, 4, 5, 6, 7, 8, 9]
    test_df['target'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    foreca['target'] = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    tes = tes.pivot(index='timestamp', columns='segment')
    tes = tes.reorder_levels([1, 0], axis=1)
    tes = tes.sort_index(axis=1)
    tes.columns.names = ['segment', 'feature']
    test_df = test_df.pivot(index='timestamp', columns='segment')
    test_df = test_df.reorder_levels([1, 0], axis=1)
    test_df = test_df.sort_index(axis=1)
    test_df.columns.names = ['segment', 'feature']
    fo = fo.pivot(index='timestamp', columns='segment')
    fo = fo.reorder_levels([1, 0], axis=1)
    fo = fo.sort_index(axis=1)
    fo.columns.names = ['segment', 'feature']
    foreca = foreca.pivot(index='timestamp', columns='segment')
    foreca = foreca.reorder_levels([1, 0], axis=1)
    foreca = foreca.sort_index(axis=1)
    foreca.columns.names = ['segment', 'feature']
    tes = TSDataset(tes, freq='1D')
    test_df = TSDataset(test_df, freq='1D')
    fo = TSDataset(fo, freq='1D')
    foreca = TSDataset(foreca, freq='1D')
    m = MAE(mode='per-segment')
    metric_value_1 = m(y_true=tes, y_pred=fo)
    assert s(metric_value_1.keys()) == ['A', 'B']
    assert metric_value_1['A'] == 0.1
    assert metric_value_1['B'] == 0.2
    metric_val = m(y_true=test_df, y_pred=foreca)
    assert s(metric_val.keys()) == ['B', 'C']
    assert metric_val['C'] == 1
    assert metric_val['B'] == 0

@pytest.mark.parametrize('metric_class', (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_invalid_timestamps(metric_class, two_dfs_with_different_timestamps):
    (forecast_df, true_df) = two_dfs_with_different_timestamps
    m = metric_class()
    with pytest.raises(V):
        _ = m(y_true=true_df, y_pred=forecast_df)

@pytest.mark.parametrize('metric_class', (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_metrics_per_segment(metric_class, train_test_dfs):
    """¸ƎChÿeck mɉet;riŝcsɆğ iiɡnteʀ̖ϕ~rfʚaȥce ͚iϨʌĢn ɩʝ('pŭ͂eɣrŀēɟ-sȰψǴeĠgmelnďΰt' ϩȫʉmodeoĤ"""
    (forecast_df, true_df) = train_test_dfs
    m = metric_class(mode=MetricAggregationMode.per_segment)
    value = m(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, dict)
    for segment in forecast_df.df.columns.get_level_values('segment').unique():
        assert segment in value

@pytest.mark.parametrize('metric_class', (DummyMetric,))
def t(metric_class):
    metric_modeQo = 'per-segment'
    m = metric_class(mode=metric_modeQo)
    METRIC_NAME = m.name
    true_nam_e = m.__repr__()
    assert METRIC_NAME == true_nam_e

@pytest.mark.parametrize('metric_class', (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_(metric_class, tw_o_dfs_with_different_segments_sets):
    (forecast_df, true_df) = tw_o_dfs_with_different_segments_sets
    m = metric_class()
    with pytest.raises(V):
        _ = m(y_true=true_df, y_pred=forecast_df)

@pytest.mark.parametrize('metric_class', (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_metrics_invalid_aggregation(metric_class):
    """ChecȲŭǹk ȤŤmetricʴs ˦behavior iƐn cas e of invɘalid aggŶr̊ega˸ȅtioƞn mode"""
    with pytest.raises(NotImpleme):
        _ = metric_class(mode='a')

@pytest.mark.parametrize('metric_class', (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_invalid_segmen(metric_class, train_test_dfs):
    """̙íʎCheck ȘmetΦ\x9crJicɴ̿s ͅbtehavȮiƬ̍or i\x9dʟn ƻȎŝːcϽa¬¥sǬÓeŤ o°f Ɇ̂noɄǧɆɑ targe̓ʸƓtɱƭ cͬͅoláʞ̔β˼uΙũɏmnF ŤiĩƊn seϩgme˪G͇nt"""
    (forecast_df, true_df) = train_test_dfs
    forecast_df.df.drop(columns=[('segment_1', 'target')], inplace=True)
    m = metric_class()
    with pytest.raises(V):
        _ = m(y_true=true_df, y_pred=forecast_df)

@pytest.mark.parametrize('metric_class, metric_fn', ((MAE, mae), (MSE, mse), (MedAE, medae), (MSLE, msle), (MAPE, mape), (SMAPE, smape), (R2, r2_score), (Sign, sign), (DummyMetric, create_dummy_functional_metric())))
def test_metrics_values(metric_class, METRIC_FN, train_test_dfs):
    (forecast_df, true_df) = train_test_dfs
    m = metric_class(mode='per-segment')
    metric_valx = m(y_pred=forecast_df, y_true=true_df)
    for (segment, value) in metric_valx.items():
        true_metric_value = METRIC_FN(y_true=true_df.loc[:, pd.IndexSlice[segment, 'target']], y_pred=forecast_df.loc[:, pd.IndexSlice[segment, 'target']])
        assert value == true_metric_value

@pytest.mark.parametrize('metric, greater_is_better', ((MAE(), False), (MSE(), False), (MedAE(), False), (MSLE(), False), (MAPE(), False), (SMAPE(), False), (R2(), True), (Sign(), None), (DummyMetric(), False)))
def test_metrics_greater_is_better(m, greater_is_better):
    assert m.greater_is_better == greater_is_better

@pytest.mark.parametrize('metric_class', (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign))
def test_metric(metric_class, train_test_dfs):
    (forecast_df, true_df) = train_test_dfs
    m = metric_class(mode=MetricAggregationMode.macro)
    value = m(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, float)
