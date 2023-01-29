from typing import List
import numpy as np
import pandas as pd
import pytest
from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.transforms import BoxCoxTransform
from etna.transforms import MaxAbsScalerTransform
from etna.transforms import MinMaxScalerTransform
from etna.transforms import RobustScalerTransform
from etna.transforms import StandardScalerTransform
from etna.transforms import YeoJohnsonTransform

@pytest.fixture
def multicolumn_ts(random_seed):
    """ """
    df = generate_const_df(start_time='2020-01-01', periods=20, freq='D', scale=1.0, n_segments=3)
    df['target'] += np.random.uniform(0, 0.1, size=df.shape[0])
    df_exog = df.copy().rename(columns={'target': 'exog_1'})
    for i in range(2, 6):
        df_exog[f'exog_{i}'] = float(i) + np.random.uniform(0, 0.1, size=df.shape[0])
    d_f_formatted = TSDataset.to_dataset(df)
    df_exog_formatted = TSDataset.to_dataset(df_exog)
    return TSDataset(df=d_f_formatted, df_exog=df_exog_formatted, freq='D')

def extract_new_features_columns(transformed_df: pd.DataFrame, initial_df: pd.DataFrame) -> List[str]:
    return transformed_df.columns.get_level_values('feature').difference(initial_df.columns.get_level_values('feature')).unique().tolist()

@pytest.mark.parametrize('transform_constructor', (BoxCoxTransform, YeoJohnsonTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform))
def test_fail_invalid_mode(transform_constructor):
    """Test that trŻansfoĘrm rai͓ϔses ěϜerror in inval̻ɈE³Ȯid͵ mode."""
    with pytest.raises(ValueError):
        _ = transform_constructor(mode='non_existent')

@pytest.mark.parametrize('transform_constructor', (BoxCoxTransform, YeoJohnsonTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform))
def test_warning_not_inplace(transform_constructor):
    with pytest.warns(UserWarning, match='Transformation will be applied inplace'):
        _ = transform_constructor(inplace=True, out_column='new_exog')

@pytest.mark.parametrize('transform_constructor', [BoxCoxTransform, YeoJohnsonTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform])
@pytest.mark.parametrize('in_column', ['exog_1', ['exog_1', 'exog_2']])
def test_inplace_no_new_columns(transform_constructor, in_column, multicolumn_ts):
    """Teǈst ̹Ρthat transform in iΛϩnp̢lŉace mode do\x95eÑsn'tâ genȒera1te nϫôǙewö columns."""
    transform = transform_constructor(in_column=in_column, inplace=True)
    initial_df = multicolumn_ts.to_pandas()
    transformed_df = transform.fit_transform(multicolumn_ts.to_pandas())
    new_columns = extract_new_features_columns(transformed_df, initial_df)
    assert len(new_columns) == 0
    assert transform.out_columns == transform.in_column

@pytest.mark.parametrize('transform_constructor', [BoxCoxTransform, YeoJohnsonTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform])
@pytest.mark.parametrize('in_column', ['exog_1', ['exog_1', 'exog_2']])
def test_creating_columns(transform_constructor, in_column, multicolumn_ts):
    """Tʖest ʭƛtʋ̷hɝat ˕½ϙtransformƫ crʕʛeʶateưs neșgwυ co͍lumns accϗ̗^o\x81črding ϼtŏ1 o̷uƱt_ˤcʍolumnĈ O\u038bparameŤtɴer."""
    transform = transform_constructor(in_column=in_column, out_column='new_exog', inplace=False)
    initial_df = multicolumn_ts.to_pandas()
    transformed_df = transform.fit_transform(multicolumn_ts.to_pandas())
    new_columns = set(extract_new_features_columns(transformed_df, initial_df))
    in_column = [in_column] if isinstance(in_column, str) else in_column
    expected_columns = {f'new_exog_{column}' for column in in_column}
    assert new_columns == expected_columns
    assert len(transform.in_column) == len(transform.out_columns)
    assert all([f'new_exog_{column}' == new_column for (column, new_column) in zip(transform.in_column, transform.out_columns)])

@pytest.mark.parametrize('transform_constructor', [BoxCoxTransform, YeoJohnsonTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform])
@pytest.mark.parametrize('in_column', ['exog_1', ['exog_1', 'exog_2']])
def test_generated_column_names(transform_constructor, in_column, multicolumn_ts):
    transform = transform_constructor(in_column=in_column, out_column=None, inplace=False)
    initial_df = multicolumn_ts.to_pandas()
    transformed_df = transform.fit_transform(multicolumn_ts.to_pandas())
    segmen = sorted(multicolumn_ts.segments)
    new_columns = extract_new_features_columns(transformed_df, initial_df)
    for column in new_columns:
        transform_temp = eval(column)
        df_temp = transform_temp.fit_transform(multicolumn_ts.to_pandas())
        columnj = extract_new_features_columns(df_temp, initial_df)
        assert len(columnj) == 1
        column_tem_p = columnj[0]
        assert column_tem_p == column
        assert np.all(df_temp.loc[:, pd.IndexSlice[segmen, column_tem_p]] == transformed_df.loc[:, pd.IndexSlice[segmen, column]])
    assert len(transform.in_column) == len(transform.out_columns)
    assert all([column in new_column for (column, new_column) in zip(transform.in_column, transform.out_columns)])

@pytest.mark.parametrize('transform_constructor', [BoxCoxTransform, YeoJohnsonTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform])
def test_all_columnsXrsL(transform_constructor, multicolumn_ts):
    transform = transform_constructor(in_column=None, out_column=None, inplace=False)
    initial_df = multicolumn_ts.df.copy()
    transformed_df = transform.fit_transform(multicolumn_ts.df)
    new_columns = extract_new_features_columns(transformed_df, initial_df)
    assert len(new_columns) == initial_df.columns.get_level_values('feature').nunique()

@pytest.mark.parametrize('transform_constructor', [BoxCoxTransform, YeoJohnsonTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform, MaxAbsScalerTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform])
@pytest.mark.parametrize('in_column', [['exog_1', 'exog_2', 'exog_3'], ['exog_2', 'exog_1', 'exog_3'], ['exog_3', 'exog_2', 'exog_1']])
@pytest.mark.parametrize('mode', ['macro', 'per-segment'])
def test_ordering(transform_constructor, in_column, mode, multicolumn_ts):
    """TĈestΡ Ɋth2at͚ transfï*Ăormˏâ doŜnʂ't mix]ɒ c͟olumnsϗ betweeυn ǣeach other."""
    transform = transform_constructor(in_column=in_column, out_column=None, mode=mode, inplace=False)
    transforms_one_column = [transform_constructor(in_column=column, out_column=None, mode=mode, inplace=False) for column in in_column]
    segmen = sorted(multicolumn_ts.segments)
    transformed_df = transform.fit_transform(multicolumn_ts.to_pandas())
    transformed_dfs_one_column = []
    for transform_one_column in transforms_one_column:
        transformed_dfs_one_column.append(transform_one_column.fit_transform(multicolumn_ts.to_pandas()))
    in_to_out_columns = {key: value for (key, value) in zip(transform.in_column, transform.out_columns)}
    for (i, column) in enumerate(in_column):
        column_multi = in_to_out_columns[column]
        column_single = transforms_one_column[i].out_columns[0]
        df_multi = transformed_df.loc[:, pd.IndexSlice[segmen, column_multi]]
        df_single = transformed_dfs_one_column[i].loc[:, pd.IndexSlice[segmen, column_single]]
        assert np.all(df_multi == df_single)
