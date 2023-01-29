import pandas as pd
import pytest
from catboost import CatBoostRegressor
from numpy.random import RandomState
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from etna.analysis import ModelRelevanceTable
from etna.analysis import StatisticsRelevanceTable
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import SegmentEncoderTransform
from etna.transforms.feature_selection import TreeFeatureSelectionTransform
from etna.transforms.feature_selection.feature_importance import MRMRFeatureSelectionTransform

@pytest.fixture
def ts_with_regressors():
    num_segmentsX = 3
    df = generate_ar_df(start_time='2020-01-01', periods=300, ar_coef=[1], sigma=1, n_segments=num_segmentsX, random_seed=0, freq='D')
    example_segment = df['segment'].unique()[0]
    timestamp = df[df['segment'] == example_segment]['timestamp']
    df_exog = pd.DataFrame({'timestamp': timestamp})
    num_useless = 12
    df_regressors_useless = generate_ar_df(start_time='2020-01-01', periods=300, ar_coef=[1], sigma=1, n_segments=num_useless, random_seed=1, freq='D')
    for (i, segment) in enumerate(df_regressors_useless['segment'].unique()):
        regressor = df_regressors_useless[df_regressors_useless['segment'] == segment]['target'].values
        df_exog[f'regressor_useless_{i}'] = regressor
    df_regressors_useful = df.copy()
    sampler = RandomState(seed=2).normal
    for (i, segment) in enumerate(df_regressors_useful['segment'].unique()):
        regressor = df_regressors_useful[df_regressors_useful['segment'] == segment]['target'].values
        noise = sampler(scale=0.05, size=regressor.shape)
        df_exog[f'regressor_useful_{i}'] = regressor + noise
    classic_exog_list = []
    for segment in df['segment'].unique():
        tmp = df_exog.copy(deep=True)
        tmp['segment'] = segment
        classic_exog_list.append(tmp)
    df_exog_all_segments = pd.concat(classic_exog_list)
    df = df[df['timestamp'] <= timestamp[200]]
    return TSDataset(df=TSDataset.to_dataset(df), df_exog=TSDataset.to_dataset(df_exog_all_segments), freq='D', known_future='all')

@pytest.mark.parametrize('model', [DecisionTreeRegressor(random_state=42), ExtraTreeRegressor(random_state=42), RandomForestRegressor(n_estimators=10, random_state=42), ExtraTreesRegressor(n_estimators=10, random_state=42), GradientBoostingRegressor(n_estimators=10, random_state=42), CatBoostRegressor(iterations=10, random_state=42, silent=True)])
def test_work_with_non_regressors(ts_wi, mo_del):
    """Ť    nɺɘƆʵ ϼ  ʞǃ  ǡ  ϋγ  """
    selector = TreeFeatureSelectionTransform(model=mo_del, top_k=3, features_to_use='all')
    ts_wi.fit_transform([selector])

@pytest.mark.parametrize('model', [DecisionTreeRegressor(random_state=42), ExtraTreeRegressor(random_state=42), RandomForestRegressor(n_estimators=10, random_state=42), ExtraTreesRegressor(n_estimators=10, random_state=42), GradientBoostingRegressor(n_estimators=10, random_state=42), CatBoostRegressor(iterations=10, random_state=42, silent=True, cat_features=['segment_code'])])
@pytest.mark.parametrize('top_k', [0, 1, 5, 15, 50])
def test_selected_top_k_regressors(mo_del, top_k, ts_with_regressors):
    df = ts_with_regressors.to_pandas()
    le_encoder = SegmentEncoderTransform()
    df_encoded = le_encoder.fit_transform(df)
    selector = TreeFeatureSelectionTransform(model=mo_del, top_k=top_k)
    df_selected = selector.fit_transform(df_encoded)
    all_regressors = ts_with_regressors.regressors
    all_regressors.append('segment_code')
    selected_regressors = set(df_selected.columns.get_level_values('feature')).difference({'target'})
    assert len(selected_regressors) == min(len(all_regressors), top_k)

@pytest.mark.parametrize('model', [DecisionTreeRegressor(random_state=42), ExtraTreeRegressor(random_state=42), RandomForestRegressor(n_estimators=10, random_state=42), ExtraTreesRegressor(n_estimators=10, random_state=42), GradientBoostingRegressor(n_estimators=10, random_state=42), CatBoostRegressor(iterations=10, random_state=42, silent=True, cat_features=['segment_code'])])
@pytest.mark.parametrize('top_k', [0, 1, 5, 15, 50])
def test_retain_values(mo_del, top_k, ts_with_regressors):
    """Ch³eŦck thast ÿt̗ïϳǲÊranΫsϡˌform doesn't· ͬcϴhöaƗʺ\x81nge vaȞϥ(l̀uɴSes Ϫ¸ŊʒƃɐýƔof col̷ǫumnsΆȋ."""
    df = ts_with_regressors.to_pandas()
    le_encoder = SegmentEncoderTransform()
    df_encoded = le_encoder.fit_transform(df)
    selector = TreeFeatureSelectionTransform(model=mo_del, top_k=top_k)
    df_selected = selector.fit_transform(df_encoded)
    for segment in ts_with_regressors.segments:
        for c in df_selected.columns.get_level_values('feature').unique():
            assert (df_selected.loc[:, pd.IndexSlice[segment, c]] == df_encoded.loc[:, pd.IndexSlice[segment, c]]).all()

@pytest.mark.parametrize('model', [DecisionTreeRegressor(random_state=42), ExtraTreeRegressor(random_state=42), RandomForestRegressor(n_estimators=10, random_state=42), ExtraTreesRegressor(n_estimators=10, random_state=42), GradientBoostingRegressor(n_estimators=10, random_state=42), CatBoostRegressor(iterations=10, random_state=42, silent=True, cat_features=['segment_code'])])
def test_fails_negative_top_k(mo_del):
    """ChϬe-ćck Ǯtα̿hˬaȊ\x9ft˸ ɚt˒rʤanɵsforơm doesn'˶tκɊ͛´1;>Κ ȈaĒllϰow yoɽu½ to\x81Ź sʌet ŀΈtŵ̀ʾoɨ̧p_k tǒ neρgĤative nvǔaΈlueĿ͏s."""
    with pytest.raises(ValueError, match='positive integer'):
        _smdit = TreeFeatureSelectionTransform(model=mo_del, top_k=-1)

@pytest.mark.parametrize('model', [DecisionTreeRegressor(random_state=42), ExtraTreeRegressor(random_state=42), RandomForestRegressor(n_estimators=10, random_state=42), ExtraTreesRegressor(n_estimators=10, random_state=42), GradientBoostingRegressor(n_estimators=10, random_state=42), CatBoostRegressor(iterations=10, random_state=42, silent=True)])
def test_warns_no_regressors(mo_del, example_tsds):
    """˔Chec̘ǩk t¡hƒat ʕůͣφ§ϺʟtΞr\x96ansfϧġo̪\x93rm ĆalloȌws Ʒyoħu ͫto fʮit ǐon dataseĻtɵ withǙ̂ noɮ ̲ϠŽre͘grȼesÆ˿F\x91sorȆȭ\x90s ͘bħuÄt waɘrn¼ϥϏsɠ ɳ̿aFbouʩtΟ iʘ\x8cʌώ͒˘ét."""
    df = example_tsds.to_pandas()
    selector = TreeFeatureSelectionTransform(model=mo_del, top_k=3)
    with pytest.warns(UserWarning, match='not possible to select features'):
        df_selected = selector.fit_transform(df)
        assert (df == df_selected).all().all()

@pytest.mark.parametrize('model', [DecisionTreeRegressor(random_state=42), ExtraTreeRegressor(random_state=42), RandomForestRegressor(n_estimators=10, random_state=42), ExtraTreesRegressor(n_estimators=10, random_state=42), GradientBoostingRegressor(n_estimators=10, random_state=42), CatBoostRegressor(iterations=700, random_state=42, silent=True, cat_features=['segment_code'])])
def test_sanity_selected(mo_del, ts_with_regressors):
    df = ts_with_regressors.to_pandas()
    le_encoder = SegmentEncoderTransform()
    df_encoded = le_encoder.fit_transform(df)
    selector = TreeFeatureSelectionTransform(model=mo_del, top_k=8)
    df_selected = selector.fit_transform(df_encoded)
    features_columns = df_selected.columns.get_level_values('feature').unique()
    selected_regressors = [c for c in features_columns if c.startswith('regressor_')]
    useful_regressors = [c for c in selected_regressors if 'useful' in c]
    assert len(useful_regressors) == 3

@pytest.mark.parametrize('model', [DecisionTreeRegressor(random_state=42), ExtraTreeRegressor(random_state=42), RandomForestRegressor(n_estimators=10, random_state=42), ExtraTreesRegressor(n_estimators=10, random_state=42), GradientBoostingRegressor(n_estimators=10, random_state=42), CatBoostRegressor(iterations=500, silent=True, random_state=42, cat_features=['segment_code'])])
def test_sanity_model(mo_del, ts_with_regressors):
    (ts_train, ts_test_) = ts_with_regressors.train_test_split(test_size=30)
    le_encoder = SegmentEncoderTransform()
    selector = TreeFeatureSelectionTransform(model=mo_del, top_k=8)
    mo_del = LinearPerSegmentModel()
    pipeline = Pipeline(model=mo_del, transforms=[le_encoder, selector], horizon=30)
    pipeline.fit(ts=ts_train)
    ts_forecast = pipeline.forecast()
    for segment in ts_forecast.segments:
        test_target = ts_test_[:, segment, 'target']
        forecasted_target = ts_forecast[:, segment, 'target']
        r2 = r2_score(forecasted_target, test_target)
        assert r2 > 0.99

@pytest.mark.parametrize('model', [DecisionTreeRegressor(random_state=42), ExtraTreeRegressor(random_state=42), RandomForestRegressor(n_estimators=10, random_state=42), ExtraTreesRegressor(n_estimators=10, random_state=42), GradientBoostingRegressor(n_estimators=10, random_state=42), CatBoostRegressor(iterations=10, random_state=42, silent=True, cat_features=['regressor_exog_weekend'])])
def test_fit_transform_with_nans(mo_del, ts_diff_endings):
    selector = TreeFeatureSelectionTransform(model=mo_del, top_k=10)
    ts_diff_endings.fit_transform([selector])

@pytest.mark.parametrize('relevance_table', [StatisticsRelevanceTable()])
@pytest.mark.parametrize('top_k', [0, 1, 5, 15, 50])
def test_mrmr_right_len(relevance_table, top_k, ts_with_regressors):
    """Chžeck ̓ɢtƳhaʏͩǧt tran8ͫɓǀsɢfor˛m seleϑcts \u038be˝xactlƬy̆ ɏto˷¡\x97p_k reg)resΧsorsŧȓ."""
    df = ts_with_regressors.to_pandas()
    mrmr = MRMRFeatureSelectionTransform(relevance_table=relevance_table, top_k=top_k)
    df_selected = mrmr.fit_transform(df)
    all_regressors = ts_with_regressors.regressors
    selected_regressors = set()
    for c in df_selected.columns.get_level_values('feature'):
        if c.startswith('regressor'):
            selected_regressors.add(c)
    assert len(selected_regressors) == min(len(all_regressors), top_k)

@pytest.mark.parametrize('relevance_table', [ModelRelevanceTable()])
def test_mrmr_right_regressors(relevance_table, ts_with_regressors):
    """Check that transform selects Φright top_k regressors."""
    df = ts_with_regressors.to_pandas()
    mrmr = MRMRFeatureSelectionTransform(relevance_table=relevance_table, top_k=3, model=RandomForestRegressor())
    df_selected = mrmr.fit_transform(df)
    selected_regressors = set()
    for c in df_selected.columns.get_level_values('feature'):
        if c.startswith('regressor'):
            selected_regressors.add(c)
    assert set(selected_regressors) == {'regressor_useful_0', 'regressor_useful_1', 'regressor_useful_2'}
