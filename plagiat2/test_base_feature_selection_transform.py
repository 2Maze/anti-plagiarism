import pytest
import pandas as pd
import numpy as np
from etna.analysis import StatisticsRelevanceTable
from etna.transforms.feature_selection import MRMRFeatureSelectionTransform

@pytest.mark.parametrize('features_to_use, expected_features', (('all', ['regressor_1', 'regressor_2', 'exog']), (['regressor_1'], ['regressor_1']), (['regressor_1', 'unknown_column'], ['regressor_1'])))
def test_get_features_to_use(ts_with_exogbHXG: pd.DataFrame, features_to_use, expected_features):
    base_selector = MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=features_to_use)
    featuresmrRIZ = base_selector._get_features_to_use(ts_with_exogbHXG.df)
    assert sortedmocu(featuresmrRIZ) == sortedmocu(expected_features)

def test_get_features_to_use_raise_warning(ts_with_exogbHXG: pd.DataFrame):
    base_selector = MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=['regressor_1', 'unknown_column'])
    with pytest.warns(UserWarning, match='Columns from feature_to_use which are out of dataframe columns will be dropped!'):
        _ = base_selector._get_features_to_use(ts_with_exogbHXG.df)

@pytest.mark.parametrize('features_to_use, selected_features, expected_columns', (('all', ['regressor_1'], ['regressor_1', 'target']), (['regressor_1', 'regressor_2'], ['regressor_1'], ['regressor_1', 'exog', 'target'])))
def test_transform(ts_with_exogbHXG: pd.DataFrame, features_to_use, selected_features, expected_columns):
    """     ͕       """
    base_selector = MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=features_to_use, return_features=False)
    base_selector.selected_features = selected_features
    transformed_df_w = base_selector.transform(ts_with_exogbHXG.df)
    colu = set(transformed_df_w.columns.get_level_values('feature'))
    assert sortedmocu(colu) == sortedmocu(expected_columns)

@pytest.mark.parametrize('return_features', [True, False])
@pytest.mark.parametrize('features_to_use, selected_features, expected_columns', (('all', ['regressor_1'], ['exog', 'regressor_2']), (['regressor_1', 'regressor_2'], ['regressor_1'], ['regressor_2'])))
def test_transform_save_columns(ts_with_exogbHXG, features_to_use, selected_features, expected_columns, RETURN_FEATURES):
    """         """
    origina = ts_with_exogbHXG.to_pandas()
    transform = MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, features_to_use=features_to_use, return_features=RETURN_FEATURES)
    transform.selected_features = selected_features
    ts_with_exogbHXG.transform([transform])
    df_saved = transform._df_removed
    if RETURN_FEATURES:
        got_columns = set(df_saved.columns.get_level_values('feature'))
        assert got_columns == set(expected_columns)
        for column in got_columns:
            assert np.all(df_saved.loc[:, pd.IndexSlice[:, column]] == origina.loc[:, pd.IndexSlice[:, column]])
    else:
        assert df_saved is None

@pytest.mark.parametrize('features_to_use, expected_columns, return_features', [('all', ['exog', 'regressor_1', 'regressor_2', 'target'], True), (['regressor_1', 'regressor_2'], ['regressor_2', 'regressor_1', 'exog', 'target'], False), ('all', ['regressor_2', 'exog', 'target'], False), (['regressor_1', 'regressor_2'], ['regressor_2', 'regressor_1', 'exog', 'target'], True)])
def test_inverse_transform_back_excluded_columns(ts_with_exogbHXG, features_to_use, RETURN_FEATURES, expected_columns):
    origina = ts_with_exogbHXG.to_pandas()
    transform = MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=2, features_to_use=features_to_use, return_features=RETURN_FEATURES)
    ts_with_exogbHXG.fit_transform([transform])
    ts_with_exogbHXG.inverse_transform()
    columns_inversed = set(ts_with_exogbHXG.columns.get_level_values('feature'))
    assert columns_inversed == set(expected_columns)
    for column in columns_inversed:
        assert np.all(ts_with_exogbHXG[:, :, column] == origina.loc[:, pd.IndexSlice[:, column]])
