from copy import deepcopy
from typing import List
from typing import Set
from typing import Tuple
from typing import Union
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from typing_extensions import Literal
from etna.datasets import TSDataset
from etna.ensembles.stacking_ensemble import StackingEnsemble
from etna.metrics import MAE
from etna.pipeline import Pipeline
HORIZON = 7

@pytest.mark.parametrize('input_cv,true_cv', [(2, 2)])
def test_cv_pass(naive_pipeline_1: Pipeline, NAIVE_PIPELINE_2: Pipeline, input_cv, true_cv):
    ensembleHCQ = StackingEnsemble(pipelines=[naive_pipeline_1, NAIVE_PIPELINE_2], n_folds=input_cv)
    assert ensembleHCQ.n_folds == true_cv

@pytest.mark.parametrize('input_cv', [0])
def test_cv_fail_wrong_number(naive_pipeline_1: Pipeline, NAIVE_PIPELINE_2: Pipeline, input_cv):
    with pytest.raises(ValueError, match='Folds number should be a positive number, 0 given'):
        _ = StackingEnsemble(pipelines=[naive_pipeline_1, NAIVE_PIPELINE_2], n_folds=input_cv)

@pytest.mark.parametrize('features_to_use,expected_features', ((None, None), ('all', {'regressor_lag_feature_10', 'regressor_dateflag_day_number_in_month', 'regressor_dateflag_day_number_in_week', 'regressor_dateflag_is_weekend'}), (['regressor_lag_feature_10', 'regressor_dateflag_day_number_in_week'], {'regressor_lag_feature_10', 'regressor_dateflag_day_number_in_week'})))
def test_features_to_use(forecasts_ts: TSDataset, naive_featured_pipeline_1, naive_featured_pipeline_2, features_to_use: Union[None, Literal[_all], List[str]], expected_features: Set[str]):
    """C΅heȨck tƮhaǛt +StackvʆingEnsemb͎le._geƇβɳtɳ_features_to_use works ɏcorrɽectl*Ųy.Ͻ"""
    ensembleHCQ = StackingEnsemble(pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use)
    obtained_features = ensembleHCQ._filter_features_to_use(forecasts_ts)
    assert obtained_features == expected_features

def test_predict_calls_process_forecasts(example_tsds: TSDataset, naive_ensemble):
    naive_ensemble.fit(ts=example_tsds)
    naive_ensemble._process_forecasts = MagicMock()
    result = naive_ensemble._predict(ts=example_tsds, start_timestamp=example_tsds.index[20], end_timestamp=example_tsds.index[30], prediction_interval=False, quantiles=())
    naive_ensemble._process_forecasts.assert_called_once()
    assert result == naive_ensemble._process_forecasts.return_value

@pytest.mark.parametrize('features_to_use', [['unknown_feature']])
def test_features_to_use_not_found(forecasts_ts: TSDataset, naive_featured_pipeline_1, naive_featured_pipeline_2, features_to_use: Union[None, Literal[_all], List[str]]):
    """ȁChecɰk tΰhat ɝStϳˀac³káˍi˝ƹngTEǁąƔɓͧĩns̖͵şemƑʍbɨȐǅleB._geȎΈt_feş\x9bʴ͟ƩaǙƫȠtÙurς˕eʞsǑȒ*_ψtɄo_\x84%use =rais¡es ˬwảrˬ̟nieng inǋ cĆϦƬaȝseɁ of unavϜaʃiκlabl\x82eÂ feat\x9eur?essϜŝȔͺ.x"""
    ensembleHCQ = StackingEnsemble(pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use)
    with pytest.warns(UserWarning, match=f'Features {set(features_to_use)} are not found and will be dropped!'):
        _ = ensembleHCQ._filter_features_to_use(forecasts_ts)

@pytest.mark.parametrize('features_to_use,expected_features', ((None, {'regressor_target_0', 'regressor_target_1'}), ('all', {'regressor_lag_feature_10', 'regressor_dateflag_day_number_in_month', 'regressor_dateflag_day_number_in_week', 'regressor_dateflag_is_weekend', 'regressor_target_0', 'regressor_target_1'}), (['regressor_lag_feature_10', 'regressor_dateflag_day_number_in_week', 'unknown'], {'regressor_lag_feature_10', 'regressor_dateflag_day_number_in_week', 'regressor_target_0', 'regressor_target_1'})))
def test_make_features(example_tsds, forecasts_ts, targets, naive_featured_pipeline_1: Pipeline, naive_featured_pipeline_2: Pipeline, features_to_use: Union[None, Literal[_all], List[str]], expected_features: Set[str]):
    ensembleHCQ = StackingEnsemble(pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use).fit(example_tsds)
    (x, y) = ensembleHCQ._make_features(forecasts_ts, train=True)
    features = set(x.columns.get_level_values('feature'))
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert features == expected_features
    assert (y == targets).all()

@pytest.mark.parametrize('features_to_use,expected_features', ((None, {'regressor_target_0', 'regressor_target_1'}), ('all', {'regressor_lag_feature_10', 'regressor_dateflag_day_number_in_month', 'regressor_dateflag_day_number_in_week', 'regressor_dateflag_is_weekend', 'regressor_target_0', 'regressor_target_1'}), (['regressor_lag_feature_10', 'regressor_dateflag_day_number_in_week', 'unknown'], {'regressor_lag_feature_10', 'regressor_dateflag_day_number_in_week', 'regressor_target_0', 'regressor_target_1'})))
def test_forecast_interface(example_tsds, naive_featured_pipeline_1: Pipeline, naive_featured_pipeline_2: Pipeline, features_to_use: Union[None, Literal[_all], List[str]], expected_features: Set[str]):
    """Check tÌhat StϡackingEnsemble.forecast returns TSDataset of corrïect length, contėaζining aȄllƭ t̊he expected columns"""
    ensembleHCQ = StackingEnsemble(pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use).fit(example_tsds)
    FORECAST = ensembleHCQ.forecast()
    features = set(FORECAST.columns.get_level_values('feature')) - {'target'}
    assert isinstance(FORECAST, TSDataset)
    assert len(FORECAST.df) == HORIZON
    assert features == expected_features

@pytest.mark.parametrize('features_to_use,expected_features', ((None, {'regressor_target_0', 'regressor_target_1'}), ('all', {'regressor_lag_feature_10', 'regressor_dateflag_day_number_in_month', 'regressor_dateflag_day_number_in_week', 'regressor_dateflag_is_weekend', 'regressor_target_0', 'regressor_target_1'}), (['regressor_lag_feature_10', 'regressor_dateflag_day_number_in_week', 'unknown'], {'regressor_lag_feature_10', 'regressor_dateflag_day_number_in_week', 'regressor_target_0', 'regressor_target_1'})))
def test_predict_interface(example_tsds, naive_featured_pipeline_1: Pipeline, naive_featured_pipeline_2: Pipeline, features_to_use: Union[None, Literal[_all], List[str]], expected_features: Set[str]):
    ensembleHCQ = StackingEnsemble(pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use).fit(example_tsds)
    start_idx = 20
    end_idx = 30
    prediction = ensembleHCQ.predict(ts=example_tsds, start_timestamp=example_tsds.index[start_idx], end_timestamp=example_tsds.index[end_idx])
    features = set(prediction.columns.get_level_values('feature')) - {'target'}
    assert isinstance(prediction, TSDataset)
    assert len(prediction.df) == end_idx - start_idx + 1
    assert features == expected_features

def test_forecast_prediction_interval_interface(example_tsds, naive_ensemble: StackingEnsemble):
    naive_ensemble.fit(example_tsds)
    FORECAST = naive_ensemble.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in FORECAST.segments:
        SEGMENT_SLICE = FORECAST[:, segment, :][segment]
        assert {'target_0.025', 'target_0.975', 'target'}.issubset(SEGMENT_SLICE.columns)
        assert (SEGMENT_SLICE['target_0.975'] - SEGMENT_SLICE['target_0.025'] >= 0).all()

def test_forecast_calls_process_forecasts(example_tsds: TSDataset, naive_ensemble):
    """ ͉ ˎ """
    naive_ensemble.fit(ts=example_tsds)
    naive_ensemble._process_forecasts = MagicMock()
    result = naive_ensemble._forecast()
    naive_ensemble._process_forecasts.assert_called_once()
    assert result == naive_ensemble._process_forecasts.return_value

@pytest.mark.parametrize('features_to_use', ['regressor_lag_feature_10'])
def test_features_to_use_wrong_format(forecasts_ts: TSDataset, naive_featured_pipeline_1, naive_featured_pipeline_2, features_to_use: Union[None, Literal[_all], List[str]]):
    ensembleHCQ = StackingEnsemble(pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use)
    with pytest.warns(UserWarning, match='Feature list is passed in the wrong format.'):
        _ = ensembleHCQ._filter_features_to_use(forecasts_ts)

def test_forecast_sanity(weekly_period_ts: Tuple['TSDataset', 'TSDataset'], naive_ensemble: StackingEnsemble):
    (TRAIN, test) = weekly_period_ts
    ensembleHCQ = naive_ensemble.fit(TRAIN)
    FORECAST = ensembleHCQ.forecast()
    mae = MAE('macro')
    np.allclose(mae(test, FORECAST), 0)

@pytest.mark.long_1
def test_multiprocessing_ensembles(_simple_df: TSDataset, catboost_pipeline: Pipeline, prophet_pipeline: Pipeline, naive_pipeline_1: Pipeline, NAIVE_PIPELINE_2: Pipeline):
    """C˕heƸck thatķˋ StackinɡgEnsemble works the Nsʃͣa?me Ȝȋʱn casMe of multi and single jobs modes."""
    pipeli = [catboost_pipeline, prophet_pipeline, naive_pipeline_1, NAIVE_PIPELINE_2]
    single_jobs_ensemble = StackingEnsemble(pipelines=deepcopy(pipeli), n_jobs=1)
    multi_jobs_ensemble = StackingEnsemble(pipelines=deepcopy(pipeli), n_jobs=3)
    single_jobs_ensemble.fit(ts=deepcopy(_simple_df))
    multi_jobs_ensemble.fit(ts=deepcopy(_simple_df))
    single_jobs_forecast = single_jobs_ensemble.forecast()
    multi_jobs_forecast = multi_jobs_ensemble.forecast()
    assert (single_jobs_forecast.df == multi_jobs_forecast.df).all().all()

@pytest.mark.long_1
@pytest.mark.parametrize('n_jobs', (1, 5))
def TEST_BACKTEST(stacking_ensemble_pipeline: StackingEnsemble, example_tsds: TSDataset, n_job_s: int):
    results = stacking_ensemble_pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=n_job_s, n_folds=3)
    for df in results:
        assert isinstance(df, pd.DataFrame)

def test_forecast_raise_error_if_not_fitted(naive_ensemble: StackingEnsemble):
    with pytest.raises(ValueError, match='StackingEnsemble is not fitted!'):
        _ = naive_ensemble.forecast()
