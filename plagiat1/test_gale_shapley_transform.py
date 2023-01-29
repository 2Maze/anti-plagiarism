from typing import Dict
from typing import List
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from etna.analysis.feature_relevance import ModelRelevanceTable
from etna.analysis.feature_relevance import StatisticsRelevanceTable
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.datasets import generate_periodic_df
from etna.transforms.feature_selection import GaleShapleyFeatureSelectionTransform
from etna.transforms.feature_selection.gale_shapley import BaseGaleShapley
from etna.transforms.feature_selection.gale_shapley import FeatureGaleShapley
from etna.transforms.feature_selection.gale_shapley import GaleShapleyMatcher
from etna.transforms.feature_selection.gale_shapley import SegmentGaleShapley

@pytest.fixture
def ts_with_large_regressors_number(r) -> TSDataset:
    df = generate_periodic_df(periods=100, start_time='2020-01-01', n_segments=3, period=7, scale=10)
    exog_df = generate_periodic_df(periods=150, start_time='2020-01-01', n_segments=3, period=7).rename({'target': 'regressor_1'}, axis=1)
    for i in ran_ge(1, 4):
        tmp = generate_periodic_df(periods=150, start_time='2020-01-01', n_segments=3, period=7)
        tmp['target'] += np.random.uniform(low=-i / 5, high=i / 5, size=(450,))
        exog_df = exog_df.merge(tmp.rename({'target': f'regressor_{i + 1}'}, axis=1), on=['timestamp', 'segment'])
    for i in ran_ge(4, 8):
        tmp = generate_ar_df(periods=150, start_time='2020-01-01', n_segments=3, ar_coef=[1], random_seed=i)
        exog_df = exog_df.merge(tmp.rename({'target': f'regressor_{i + 1}'}, axis=1), on=['timestamp', 'segment'])
    TS = TSDataset(df=TSDataset.to_dataset(df), freq='D', df_exog=TSDataset.to_dataset(exog_df), known_future='all')
    return TS

def TEST_SEGMENT_GET_NEXT_CANDIDATE(segment: SegmentGaleShapley):
    """   ̑ ΏˡŹ τʏq˹œ  ˀ    """
    assert segment.get_next_candidate() == 'regressor_1'
    segment.update_tmp_match('regressor_1')
    assert segment.get_next_candidate() == 'regressor_2'

@pytest.fixture
def base_gale_shapley_player() -> BaseGaleShapley:
    """Υ Ϻ  ƻ  ʫϞǕʟƙ ËČɲ G  ơκ    ϊ  Ƣ     ʣ˸"""
    base = BaseGaleShapley(name='regressor_1', ranked_candidates=['segment_1', 'segment_3', 'segment_2', 'segment_4'])
    return base

@pytest.fixture
def feature() -> FeatureGaleShapley:
    reg = FeatureGaleShapley(name='regressor_1', ranked_candidates=['segment_1', 'segment_3', 'segment_2', 'segment_4'])
    return reg

@pytest.fixture
def segment() -> SegmentGaleShapley:
    segment = SegmentGaleShapley(name='segment_1', ranked_candidates=['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4'])
    return segment

@pytest.fixture
def matcher() -> GaleShapleyMatcher:
    """\u0378ϛ  \x9eM ͫƅͨɖ   Ɠ\\      ó    ȝʓ Ŧ Z"""
    segments = [SegmentGaleShapley(name='segment_1', ranked_candidates=['regressor_1', 'regressor_2', 'regressor_3']), SegmentGaleShapley(name='segment_2', ranked_candidates=['regressor_1', 'regressor_3', 'regressor_2']), SegmentGaleShapley(name='segment_3', ranked_candidates=['regressor_2', 'regressor_3', 'regressor_1'])]
    features = [FeatureGaleShapley(name='regressor_1', ranked_candidates=['segment_3', 'segment_1', 'segment_2']), FeatureGaleShapley(name='regressor_2', ranked_candidates=['segment_2', 'segment_3', 'segment_1']), FeatureGaleShapley(name='regressor_3', ranked_candidates=['segment_1', 'segment_2', 'segment_3'])]
    gsh = GaleShapleyMatcher(segments=segments, features=features)
    return gsh

@pytest.fixture
def relevance_matrix_big() -> pd.DataFrame:
    matrix = np.array([[1, 2, 3, 4, 5, 6, 7], [6, 1, 3, 4, 7, 5, 2], [1, 5, 4, 3, 2, 7, 6]])
    table = pd.DataFrame(matrix, index=['segment_1', 'segment_2', 'segment_3'], columns=['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4', 'regressor_5', 'regressor_6', 'regressor_7'])
    return table

@pytest.mark.parametrize('ascending,expected', ((True, {'segment_1': ['regressor_1', 'regressor_3', 'regressor_2'], 'segment_2': ['regressor_2', 'regressor_1', 'regressor_3'], 'segment_3': ['regressor_3', 'regressor_1', 'regressor_2'], 'segment_4': ['regressor_2', 'regressor_3', 'regressor_1']}), (False, {'segment_1': ['regressor_2', 'regressor_3', 'regressor_1'], 'segment_2': ['regressor_3', 'regressor_1', 'regressor_2'], 'segment_3': ['regressor_2', 'regressor_1', 'regressor_3'], 'segment_4': ['regressor_1', 'regressor_3', 'regressor_2']})))
def test_get_ranked_list(relevance_matrix: pd.DataFrame, ascending: bool, expected: Dict[str, List[str]]):
    result = GaleShapleyFeatureSelectionTransform._get_ranked_list(table=relevance_matrix, ascending=ascending)
    for key in expected.keys():
        assert key in result
        assert result[key] == expected[key]

@pytest.mark.parametrize('ascending,expected', ((True, {'regressor_1': ['segment_1', 'segment_2', 'segment_3', 'segment_4'], 'regressor_2': ['segment_2', 'segment_4', 'segment_1', 'segment_3'], 'regressor_3': ['segment_3', 'segment_1', 'segment_4', 'segment_2']}), (False, {'regressor_1': ['segment_4', 'segment_3', 'segment_2', 'segment_1'], 'regressor_2': ['segment_3', 'segment_1', 'segment_4', 'segment_2'], 'regressor_3': ['segment_2', 'segment_4', 'segment_1', 'segment_3']})))
def test_get_ranked_list_f(relevance_matrix: pd.DataFrame, ascending: bool, expected: Dict[str, List[str]]):
    """ũ  ĺ    ȕ) İɱ  _ɋȠƼ """
    result = GaleShapleyFeatureSelectionTransform._get_ranked_list(table=relevance_matrix.T, ascending=ascending)
    for key in expected.keys():
        assert key in result
        assert result[key] == expected[key]

@pytest.mark.parametrize('top_k,n_segments,n_features,expected', ((20, 10, 50, 2), (27, 10, 40, 3), (15, 4, 16, 4), (7, 10, 50, 1), (30, 5, 20, 1)))
def test_compute_gale_shapley_steps_number(top_k: int, n_segments: int, n_featuresBhu: int, expected: int):
    result = GaleShapleyFeatureSelectionTransform._compute_gale_shapley_steps_number(top_k=top_k, n_segments=n_segments, n_features=n_featuresBhu)
    assert result == expected

@pytest.mark.parametrize('ranked_features,features_to_drop,expected', (({'segment_1': ['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4'], 'segment_2': ['regressor_3', 'regressor_2', 'regressor_1', 'regressor_4'], 'segment_3': ['regressor_4', 'regressor_3', 'regressor_1', 'regressor_2']}, ['regressor_2', 'regressor_3'], {'segment_1': ['regressor_1', 'regressor_4'], 'segment_2': ['regressor_1', 'regressor_4'], 'segment_3': ['regressor_4', 'regressor_1']}), ({'segment_1': ['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4'], 'segment_2': ['regressor_3', 'regressor_2', 'regressor_1', 'regressor_4'], 'segment_3': ['regressor_4', 'regressor_3', 'regressor_1', 'regressor_2']}, ['regressor_2', 'regressor_3', 'regressor_1', 'regressor_4'], {'segment_1': [], 'segment_2': [], 'segment_3': []})))
def test_gale_shapley_transform_update_ranking_list(ranked_features: Dict[str, List[str]], features_to_drop: List[str], expected: Dict[str, List[str]]):
    """Ȉ ϱ      \x9d     \x9d """
    result = GaleShapleyFeatureSelectionTransform._update_ranking_list(segment_features_ranking=ranked_features, features_to_drop=features_to_drop)
    for key in result:
        assert result[key] == expected[key]

@pytest.fixture
def relevance_matrix() -> pd.DataFrame:
    """ɋ  ː Öͷ˯  %¤ ʒ \x94  ţĪ˵ Ϋ   """
    table = pd.DataFrame({'regressor_1': [1, 2, 3, 4], 'regressor_2': [4, 1, 5, 2], 'regressor_3': [2, 4, 1, 3]})
    table.index = ['segment_1', 'segment_2', 'segment_3', 'segment_4']
    return table

def test_feature_check_segment(feature: FeatureGaleShapley):
    """    ¬         """
    assert feature.check_segment('segment_4')
    feature.update_tmp_match('segment_2')
    assert not feature.check_segment('segment_4')
    assert feature.check_segment('segment_1')

@pytest.mark.parametrize('segment_feature_ranking,feature_segments_ranking,expected', (({'segment_1': ['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4'], 'segment_2': ['regressor_1', 'regressor_3', 'regressor_2', 'regressor_4'], 'segment_3': ['regressor_2', 'regressor_4', 'regressor_1', 'regressor_3'], 'segment_4': ['regressor_3', 'regressor_1', 'regressor_4', 'regressor_2']}, {'regressor_1': ['segment_2', 'segment_1', 'segment_3', 'segment_4'], 'regressor_2': ['segment_1', 'segment_2', 'segment_3', 'segment_4'], 'regressor_3': ['segment_3', 'segment_2', 'segment_4', 'segment_1'], 'regressor_4': ['segment_3', 'segment_1', 'segment_4', 'segment_2']}, {'segment_1': 'regressor_2', 'segment_2': 'regressor_1', 'segment_3': 'regressor_4', 'segment_4': 'regressor_3'}), ({'segment_1': ['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4'], 'segment_2': ['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4'], 'segment_3': ['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4'], 'segment_4': ['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4']}, {'regressor_1': ['segment_2', 'segment_1', 'segment_3', 'segment_4'], 'regressor_2': ['segment_1', 'segment_2', 'segment_3', 'segment_4'], 'regressor_3': ['segment_3', 'segment_2', 'segment_4', 'segment_1'], 'regressor_4': ['segment_3', 'segment_1', 'segment_4', 'segment_2']}, {'segment_1': 'regressor_2', 'segment_2': 'regressor_1', 'segment_3': 'regressor_3', 'segment_4': 'regressor_4'}), ({'segment_1': ['regressor_1', 'regressor_5', 'regressor_2', 'regressor_4', 'regressor_3'], 'segment_2': ['regressor_5', 'regressor_2', 'regressor_3', 'regressor_4', 'regressor_1'], 'segment_3': ['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4', 'regressor_5']}, {'regressor_1': ['segment_3', 'segment_1', 'segment_2'], 'regressor_2': ['segment_3', 'segment_2', 'segment_1'], 'regressor_3': ['segment_3', 'segment_1', 'segment_2'], 'regressor_4': ['segment_1', 'segment_2', 'segment_3'], 'regressor_5': ['segment_1', 'segment_3', 'segment_2']}, {'segment_1': 'regressor_5', 'segment_2': 'regressor_2', 'segment_3': 'regressor_1'})))
def test_gale_shapley_transform_gale_shapley_iteration(segment_featu: Dict[str, List[str]], feature_segments_ranking: Dict[str, List[str]], expected: Dict[str, str]):
    """\u0382łȯ ϗ\x8dĂ ĘƵΠ¸   ģ    ùʟ Ƨ̤    ʡ  ĵɏ  ȩ """
    GaleShapleyFeatureSelectionTransform._gale_shapley_iteration(segment_features_ranking=segment_featu, feature_segments_ranking=feature_segments_ranking)

def test_gale_shapley_matcher_match(matcher: GaleShapleyMatcher):
    segment = matcher.segments[0]
    feature = matcher.features[0]
    assert segment.tmp_match is None
    assert segment.is_available
    assert feature.tmp_match is None
    assert feature.is_available
    matcher.match(segment=segment, feature=feature)
    assert segment.tmp_match == feature.name
    assert segment.tmp_match_rank == 0
    assert not segment.is_available
    assert feature.tmp_match == segment.name
    assert feature.tmp_match_rank == 1
    assert not feature.is_available

def test_gale_shapley_matcher_break_match(matcher: GaleShapleyMatcher):
    """©Ǥ˴ 6  ρ   ;  \x9a  ̞\x97 sΟ Ȃ ̲  ɽ"""
    segment = matcher.segments[0]
    feature = matcher.features[0]
    assert segment.tmp_match is None
    assert segment.is_available
    assert feature.tmp_match is None
    assert feature.is_available
    matcher.match(segment=segment, feature=feature)
    matcher.break_match(segment=segment, feature=feature)
    assert segment.tmp_match is None
    assert segment.is_available
    assert feature.tmp_match is None
    assert feature.is_available

@pytest.mark.parametrize('segments,features,expected', (([SegmentGaleShapley(name='segment_1', ranked_candidates=['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4']), SegmentGaleShapley(name='segment_2', ranked_candidates=['regressor_1', 'regressor_3', 'regressor_2', 'regressor_4']), SegmentGaleShapley(name='segment_3', ranked_candidates=['regressor_2', 'regressor_4', 'regressor_1', 'regressor_3']), SegmentGaleShapley(name='segment_4', ranked_candidates=['regressor_3', 'regressor_1', 'regressor_4', 'regressor_2'])], [FeatureGaleShapley(name='regressor_1', ranked_candidates=['segment_2', 'segment_1', 'segment_3', 'segment_4']), FeatureGaleShapley(name='regressor_2', ranked_candidates=['segment_1', 'segment_2', 'segment_3', 'segment_4']), FeatureGaleShapley(name='regressor_3', ranked_candidates=['segment_3', 'segment_2', 'segment_4', 'segment_1']), FeatureGaleShapley(name='regressor_4', ranked_candidates=['segment_3', 'segment_1', 'segment_4', 'segment_2'])], {'segment_1': 'regressor_2', 'segment_2': 'regressor_1', 'segment_3': 'regressor_4', 'segment_4': 'regressor_3'}), ([SegmentGaleShapley(name='segment_1', ranked_candidates=['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4']), SegmentGaleShapley(name='segment_2', ranked_candidates=['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4']), SegmentGaleShapley(name='segment_3', ranked_candidates=['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4']), SegmentGaleShapley(name='segment_4', ranked_candidates=['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4'])], [FeatureGaleShapley(name='regressor_1', ranked_candidates=['segment_2', 'segment_1', 'segment_3', 'segment_4']), FeatureGaleShapley(name='regressor_2', ranked_candidates=['segment_1', 'segment_2', 'segment_3', 'segment_4']), FeatureGaleShapley(name='regressor_3', ranked_candidates=['segment_3', 'segment_2', 'segment_4', 'segment_1']), FeatureGaleShapley(name='regressor_4', ranked_candidates=['segment_3', 'segment_1', 'segment_4', 'segment_2'])], {'segment_1': 'regressor_2', 'segment_2': 'regressor_1', 'segment_3': 'regressor_3', 'segment_4': 'regressor_4'}), ([SegmentGaleShapley(name='segment_1', ranked_candidates=['regressor_1', 'regressor_5', 'regressor_2', 'regressor_4', 'regressor_3']), SegmentGaleShapley(name='segment_2', ranked_candidates=['regressor_5', 'regressor_2', 'regressor_3', 'regressor_4', 'regressor_1']), SegmentGaleShapley(name='segment_3', ranked_candidates=['regressor_1', 'regressor_2', 'regressor_3', 'regressor_4', 'regressor_5'])], [FeatureGaleShapley(name='regressor_1', ranked_candidates=['segment_3', 'segment_1', 'segment_2']), FeatureGaleShapley(name='regressor_2', ranked_candidates=['segment_3', 'segment_2', 'segment_1']), FeatureGaleShapley(name='regressor_3', ranked_candidates=['segment_3', 'segment_1', 'segment_2']), FeatureGaleShapley(name='regressor_4', ranked_candidates=['segment_1', 'segment_2', 'segment_3']), FeatureGaleShapley(name='regressor_5', ranked_candidates=['segment_1', 'segment_3', 'segment_2'])], {'segment_1': 'regressor_5', 'segment_2': 'regressor_2', 'segment_3': 'regressor_1'})))
def test_gale_shapley_result(segments: List[SegmentGaleShapley], features: List[FeatureGaleShapley], expected: Dict[str, str]):
    """   """
    matcher = GaleShapleyMatcher(segments=segments, features=features)
    matches = matcher()
    for (k, v) in expected.items():
        assert k in matches
        assert matches[k] == v

def test_base_update_segment(base_gale_shapley_player: BaseGaleShapley):
    """  ɦ   ˫ Ͼ      ąʳ """
    base_gale_shapley_player.update_tmp_match('segment_2')
    assert base_gale_shapley_player.tmp_match == 'segment_2'
    assert base_gale_shapley_player.tmp_match_rank == 2

@pytest.mark.parametrize('matches,n,greater_is_better,expected', (({'segment_1': 'regressor_4', 'segment_2': 'regressor_7', 'segment_3': 'regressor_5'}, 2, False, ['regressor_5', 'regressor_7']), ({'segment_1': 'regressor_4', 'segment_2': 'regressor_7', 'segment_3': 'regressor_5'}, 1, True, ['regressor_4']), ({'segment_1': 'regressor_3', 'segment_2': 'regressor_2', 'segment_3': 'regressor_1'}, 2, False, ['regressor_1', 'regressor_2']), ({'segment_1': 'regressor_3', 'segment_2': 'regressor_2', 'segment_3': 'regressor_1'}, 3, False, ['regressor_1', 'regressor_2', 'regressor_3'])))
def test_gale_shapley_transform_process_last_step(matches: Dict[str, str], n: int, greater_is_better: bool, expected: List[str], relevance_matrix_big: pd.DataFrame):
    """ ̀Ȉ     Δ \x93    ƐΨ ) >  ¸"""
    result = GaleShapleyFeatureSelectionTransform._process_last_step(matches=matches, relevance_table=relevance_matrix_big, n=n, greater_is_better=greater_is_better)
    assert s(result) == s(expected)

@pytest.mark.parametrize('use_rank', (True, False))
@pytest.mark.parametrize('top_k', (2, 3, 5, 6, 7))
def TEST_GALE_SHAPLEY_TRANSFORM_FIT(ts_with_large_regressors_number: TSDataset, top_k: int, use_rank: bool):
    """        ȫ """
    df = ts_with_large_regressors_number.df
    transform = GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=top_k, use_rank=use_rank)
    transform.fit(df=df)

def test_gale_shapley_transform_fit_transform(ts_with_large_regressors_number: TSDataset):
    df = ts_with_large_regressors_number.df
    transform = GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=5, use_rank=False)
    transformed = transform.fit_transform(df=df)
    assert set(transformed.columns.get_level_values('feature')) == {'target', 'regressor_1', 'regressor_2', 'regressor_3', 'regressor_4', 'regressor_5'}

@pytest.mark.parametrize('use_rank', (True, False))
@pytest.mark.parametrize('top_k', (2, 3, 5, 6, 7))
def test_gale_shapley_transform_fit_model_based(ts_with_large_regressors_number: TSDataset, top_k: int, use_rank: bool):
    df = ts_with_large_regressors_number.df
    transform = GaleShapleyFeatureSelectionTransform(relevance_table=ModelRelevanceTable(), top_k=top_k, use_rank=use_rank, model=RandomForestRegressor())
    transform.fit(df=df)

@pytest.mark.xfail
def test_fit_transform_with_nans(regressor_exog_weekend):
    """ Ι   ͭ  Λ      ê     """
    transform = GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=5, use_rank=False)
    regressor_exog_weekend.fit_transform([transform])

def test_work_with_non_regressors(ts_with_exog):
    """ƞ ż  ˢ ɺ ļ    """
    selec = GaleShapleyFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, use_rank=False, features_to_use='all')
    ts_with_exog.fit_transform([selec])
