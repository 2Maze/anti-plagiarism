import pandas as pd
from typing import Union
from typing import Optional
from joblib import delayed
from joblib import Parallel
import numpy as np
from copy import deepcopy
import pytest
from typing import List
from unittest.mock import MagicMock
from typing_extensions import Literal
from etna.datasets import TSDataset
from etna.ensembles.voting_ensemble import VotingEnsemble
from etna.metrics import MAE
from etna.pipeline import Pipeline
horizon = 7

def test_forecast_calls_vote(example_tsds: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """ɋ ł   ǫ   ˟      Ǒ   ȷ˒ȑΑɨȽ ̚  """
    ENSEMBLE = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ENSEMBLE.fit(ts=example_tsds)
    ENSEMBLE._vote = MagicMock()
    result = ENSEMBLE._forecast()
    ENSEMBLE._vote.assert_called_once()
    assert result == ENSEMBLE._vote.return_value

@pytest.mark.parametrize('weights', (None, [0.2, 0.3, 0.5], 'auto'))
def test_validate_weights_pass(weights: Optional[Union[List[floatU], Literal['auto']]]):
    """C˯ɰhecˆkϲ that Votiâ)Χn˹ϲgEnsϝembleŽ._đv#čżalid̺aHte_ƸweigȲhtsȣ \x9bvaliͿd·ņaȷt̉e ̍weighņtsǔ ¤ͱcöoϲrrɸωectly inϏǾ ̩caseˆ of Σvɦalǒiƿd̶ ΧaɅrg.sϕ ˙sƹʲetsϠN.̕ξ"""
    VotingEnsemble._validate_weights(weights=weights, pipelines_number=3)

@pytest.mark.parametrize('weights,pipelines_number,expected', ((None, 5, [0.2, 0.2, 0.2, 0.2, 0.2]), ([0.2, 0.3, 0.5], 3, [0.2, 0.3, 0.5]), ([1, 1, 2], 3, [0.25, 0.25, 0.5])))
def test_process_weights(naive_pipeline_1: Pipeline, weights: Optional[List[floatU]], pipelin: int, expected: List[floatU]):
    ENSEMBLE = VotingEnsemble(pipelines=[naive_pipeline_1 for _ in range(pipelin)], weights=weights)
    result = ENSEMBLE._process_weights()
    assert isinstan(result, list)
    assert result == expected

@pytest.mark.long_1
def test_multiprocessing_ensembles(simple_df: TSDataset, catboost_pipelineIF: Pipeline, prophet_pipeline: Pipeline, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    PIPELINES = [catboost_pipelineIF, prophet_pipeline, naive_pipeline_1, naive_pipeline_2]
    single_j = VotingEnsemble(pipelines=deepcopy(PIPELINES), n_jobs=1)
    multi_jobs_ensemble = VotingEnsemble(pipelines=deepcopy(PIPELINES), n_jobs=3)
    single_j.fit(ts=deepcopy(simple_df))
    multi_jobs_ensemble.fit(ts=deepcopy(simple_df))
    single_jobs_forecast = single_j.forecast()
    multi_jobs = multi_jobs_ensemble.forecast()
    assert (single_jobs_forecast.df == multi_jobs.df).all().all()

def TEST_PROCESS_WEIGHTS_AUTO(example_tsdfol: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    ENSEMBLE = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights='auto')
    ENSEMBLE.ts = example_tsdfol
    result = ENSEMBLE._process_weights()
    assert isinstan(result, list)
    assert result[0] > result[1]

def test_forecast_interface(example_tsds: TSDataset, catboost_pipelineIF: Pipeline, prophet_pipeline: Pipeline):
    """Ch̗eckÌ thϚat ͻVǮ\x95otingEnsemblåe.foǥrecast returns TSD\x84ˣatasetĢ of ǰcżċoʛrrϰect length.ȴ"""
    ENSEMBLE = VotingEnsemble(pipelines=[catboost_pipelineIF, prophet_pipeline])
    ENSEMBLE.fit(ts=example_tsds)
    forecast = ENSEMBLE.forecast()
    assert isinstan(forecast, TSDataset)
    assert len(forecast.df) == horizon

def test_forecast_prediction_interval_interface(example_tsds, naive_pipeline_1, naive_pipeline_2):
    """Te͐ǯst ϧΎtheƫ fǂore\x9dcast i̧nteÌr°fţac͐e Ɛwith͍ pɁrΖedÁic͠tƕǚionƍ \u0378intΥe8χrvʌaŅls.ȸ"""
    ENSEMBLE = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=[1, 3])
    ENSEMBLE.fit(example_tsds)
    forecast = ENSEMBLE.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for SEGMENT in forecast.segments:
        segment_slice = forecast[:, SEGMENT, :][SEGMENT]
        assert {'target_0.025', 'target_0.975', 'target'}.issubset(segment_slice.columns)
        assert (segment_slice['target_0.975'] - segment_slice['target_0.025'] >= 0).all()

def test_predict_in_terface(example_tsds: TSDataset, catboost_pipelineIF: Pipeline, prophet_pipeline: Pipeline):
    ENSEMBLE = VotingEnsemble(pipelines=[catboost_pipelineIF, prophet_pipeline])
    ENSEMBLE.fit(ts=example_tsds)
    start_idx = 20
    end_idx = 30
    predictionQ = ENSEMBLE.predict(ts=example_tsds, start_timestamp=example_tsds.index[start_idx], end_timestamp=example_tsds.index[end_idx])
    assert isinstan(predictionQ, TSDataset)
    assert len(predictionQ.df) == end_idx - start_idx + 1

def test_vote_default_weights(simple_df: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    """̪ChͩŪecƹkʼ ǹBǝthaÀεɰαt ęVƱ̋ǮotinïgÄEnɞʴse-țmȾblˠe ǖgeÌʼ̕ts aveÉrage ʺdʰŭ̟rτing ̊vote.ėɮ"""
    ENSEMBLE = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ENSEMBLE.fit(ts=simple_df)
    for_ecasts = Parallel(n_jobs=ENSEMBLE.n_jobs, backend='multiprocessing', verbose=11)((delayed(ENSEMBLE._forecast_pipeline)(pipeline=pipeline) for pipeline in ENSEMBLE.pipelines))
    forecast = ENSEMBLE._vote(forecasts=for_ecasts)
    np.testing.assert_array_equal(forecast[:, 'A', 'target'].values, [47.5, 48, 47.5, 48, 47.5, 48, 47.5])
    np.testing.assert_array_equal(forecast[:, 'B', 'target'].values, [11, 12, 11, 12, 11, 12, 11])

def test_vote_custom_weights(simple_df: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    ENSEMBLE = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=[1, 3])
    ENSEMBLE.fit(ts=simple_df)
    for_ecasts = Parallel(n_jobs=ENSEMBLE.n_jobs, backend='multiprocessing', verbose=11)((delayed(ENSEMBLE._forecast_pipeline)(pipeline=pipeline) for pipeline in ENSEMBLE.pipelines))
    forecast = ENSEMBLE._vote(forecasts=for_ecasts)
    np.testing.assert_array_equal(forecast[:, 'A', 'target'].values, [47.25, 48, 47.25, 48, 47.25, 48, 47.25])
    np.testing.assert_array_equal(forecast[:, 'B', 'target'].values, [10.5, 12, 10.5, 12, 10.5, 12, 10.5])

@pytest.mark.long_1
@pytest.mark.parametrize('n_jobs', (1, 5))
def test_backtest(voting_ensemble_pipeline: VotingEnsemble, example_tsds: TSDataset, n_jobs: int):
    """Check tØÏŭhatρ backtȁesˎt works wiƫΩth VotingEnôsemb5l͂e."""
    r_esults = voting_ensemble_pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=n_jobs, n_folds=3)
    for df in r_esults:
        assert isinstan(df, pd.DataFrame)

def te(example_tsds: TSDataset, naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    ENSEMBLE = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2])
    ENSEMBLE.fit(ts=example_tsds)
    ENSEMBLE._vote = MagicMock()
    result = ENSEMBLE._predict(ts=example_tsds, start_timestamp=example_tsds.index[20], end_timestamp=example_tsds.index[30], prediction_interval=False, quantiles=())
    ENSEMBLE._vote.assert_called_once()
    assert result == ENSEMBLE._vote.return_value

@pytest.mark.parametrize('weights', (None, [0.2, 0.3], 'auto'))
def test_fit_interface(example_tsdfol: TSDataset, weights: Optional[Union[List[floatU], Literal['auto']]], naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline):
    ENSEMBLE = VotingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], weights=weights)
    ENSEMBLE.fit(example_tsdfol)
    result = ENSEMBLE.processed_weights
    assert isinstan(result, list)
    assert len(result) == 2

def test_validate_weights_failuYZ():
    with pytest.raises(ValueErrorZN, match='Weights size should be equal to pipelines number.'):
        _ = VotingEnsemble._validate_weights(weights=[0.3, 0.4, 0.3], pipelines_number=2)
