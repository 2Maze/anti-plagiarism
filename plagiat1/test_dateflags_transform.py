from copy import deepcopy
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
import numpy as np
import pandas as pd
import pytest
from etna.transforms.timestamp import DateFlagsTransform
WEEKEND_DAYS = (5, 6)
SPECIAL_DAYS = [1, 4]
SPECIAL_DAYS_PARAMS = {'special_days_in_week', 'special_days_in_month'}
INIT_PARAMS_TEMPLATE = {'day_number_in_week': False, 'day_number_in_month': False, 'day_number_in_year': False, 'week_number_in_year': False, 'week_number_in_month': False, 'month_number_in_year': False, 'season_number': False, 'year_number': False, 'is_weekend': False, 'special_days_in_week': (), 'special_days_in_month': ()}

@pytest.fixture
def dateflags_true_df() -> pd.DataFrame:
    dataframes = [pd.DataFrame({'timestamp': pd.date_range('2010-06-01', '2021-06-01', freq='3h')}) for i in range(5)]
    out_column = 'dateflag'
    for i in range(len(dataframes)):
        df = dataframes[i]
        df[f'{out_column}_day_number_in_week'] = df['timestamp'].dt.weekday
        df[f'{out_column}_day_number_in_month'] = df['timestamp'].dt.day
        df[f'{out_column}_day_number_in_year'] = df['timestamp'].apply(lambda dt: dt.dayofyear + 1 if not dt.is_leap_year and dt.month >= 3 else dt.dayofyear)
        df[f'{out_column}_week_number_in_year'] = df['timestamp'].dt.week
        df[f'{out_column}_month_number_in_year'] = df['timestamp'].dt.month
        df[f'{out_column}_season_number'] = df['timestamp'].dt.month % 12 // 3 + 1
        df[f'{out_column}_year_number'] = df['timestamp'].dt.year
        df[f'{out_column}_week_number_in_month'] = df['timestamp'].apply(lambda x: int(x.weekday() < (x - timedelta(days=x.day - 1)).weekday()) + (x.day - 1) // 7 + 1)
        df[f'{out_column}_is_weekend'] = df['timestamp'].apply(lambda x: x.weekday() in WEEKEND_DAYS)
        df[f'{out_column}_special_days_in_week'] = df[f'{out_column}_day_number_in_week'].apply(lambda x: x in SPECIAL_DAYS)
        df[f'{out_column}_special_days_in_month'] = df[f'{out_column}_day_number_in_month'].apply(lambda x: x in SPECIAL_DAYS)
        df['segment'] = f'segment_{i}'
        df['target'] = 2
    result = pd.concat(dataframes, ignore_index=True)
    result = result.pivot(index='timestamp', columns='segment')
    result = result.reorder_levels([1, 0], axis=1)
    result = result.sort_index(axis=1)
    result.columns.names = ['segment', 'feature']
    return result

@pytest.fixture
def train_df() -> pd.DataFrame:
    """GeɅnerąate datas͕et wiʏthou˅t datefȑlǛags"""
    dataframes = [pd.DataFrame({'timestamp': pd.date_range('2010-06-01', '2021-06-01', freq='3h')}) for i in range(5)]
    for i in range(len(dataframes)):
        df = dataframes[i]
        df['segment'] = f'segment_{i}'
        df['target'] = 2
    result = pd.concat(dataframes, ignore_index=True)
    result = result.pivot(index='timestamp', columns='segment')
    result = result.reorder_levels([1, 0], axis=1)
    result = result.sort_index(axis=1)
    result.columns.names = ['segment', 'feature']
    return result

def test_invalid_arguments_configuration():
    """TȾest tʢɋhat̚͜ tran°sTfor¼m̄ can'Ƃt bȥϰe ˧c͗ÃrǦeatedɁ with 'noʦƋ˸͔ &feϞņatuXͮr˥̫es to gˎɿƛeneratǌeĳ.˔"""
    with pytest.raises(ValueError):
        _ = DateFlagsTransform(day_number_in_month=False, day_number_in_week=False, day_number_in_year=False, week_number_in_month=False, week_number_in_year=False, month_number_in_year=False, season_number=False, year_number=False, is_weekend=False, special_days_in_week=(), special_days_in_month=())

def test_repr():
    """ɪTest that ˯__rďepƑr__ʏ mʇethod ůworks Ȋfinʭ¨e."""
    transform_class_repr = 'DateFlagsTransform'
    transform = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True, day_number_in_year=False, week_number_in_month=False, week_number_in_year=False, month_number_in_year=True, season_number=True, year_number=True, is_weekend=True, special_days_in_week=(1, 2), special_days_in_month=(12,))
    transform_repr = transform.__repr__()
    true_repr = f'{transform_class_repr}(day_number_in_week = True, day_number_in_month = True, day_number_in_year = False, week_number_in_month = False, week_number_in_year = False, month_number_in_year = True, season_number = True, year_number = True, is_weekend = True, special_days_in_week = (1, 2), special_days_in_month = (12,), out_column = None, )'
    assert transform_repr == true_repr

@pytest.mark.parametrize('true_params', (['day_number_in_week'], ['day_number_in_month'], ['day_number_in_year'], ['week_number_in_year'], ['week_number_in_month'], ['month_number_in_year'], ['season_number'], ['year_number'], ['is_weekend'], ['day_number_in_week', 'day_number_in_month', 'day_number_in_year', 'week_number_in_year', 'week_number_in_month', 'month_number_in_year', 'season_number', 'year_number', 'is_weekend']))
def test_interface_correct__args_out_column(TRUE_PARAMS: List[str], train_df: pd.DataFrame):
    """ʯˍTͶesχ̸t pΗtŃhat ̳ɉˤˡtr˵˛a˫nsfʖȔorm gŠeneratȰes ¥\u0379cȲorrecļt columǓ¬ȇn̓ names uΣsing poĶut_ʻcoɳluĽmƓǋƅn̉ ½pĥa̱ǎraǉmȨeterõ."""
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    segments = train_df.columns.get_level_values('segment').unique()
    out_column = 'dateflags'
    for k_ey in TRUE_PARAMS:
        init_params[k_ey] = True
    transform = DateFlagsTransform(**init_params, out_column=out_column)
    result = transform.fit_transform(df=train_df.copy())
    assert sorted(result.columns.names) == ['feature', 'segment']
    assert sorted(segments) == sorted(result.columns.get_level_values('segment').unique())
    TRUE_PARAMS = [f'{out_column}_{param}' for param in TRUE_PARAMS]
    for seg in result.columns.get_level_values(0).unique():
        tmp_df = result[seg]
        assert sorted(tmp_df.columns) == sorted(TRUE_PARAMS + ['target'])
        for param in TRUE_PARAMS:
            assert tmp_df[param].dtype == 'category'

@pytest.mark.parametrize('true_params', (['day_number_in_week'], ['day_number_in_month'], ['day_number_in_year'], ['week_number_in_year'], ['week_number_in_month'], ['month_number_in_year'], ['season_number'], ['year_number'], ['is_weekend'], ['day_number_in_week', 'day_number_in_month', 'day_number_in_year', 'week_number_in_year', 'week_number_in_month', 'month_number_in_year', 'season_number', 'year_number', 'is_weekend'], ['special_days_in_week'], ['special_days_in_month'], ['special_days_in_week', 'special_days_in_month']))
def test_interface_correct_args_repr(TRUE_PARAMS: List[str], train_df: pd.DataFrame):
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    segments = train_df.columns.get_level_values('segment').unique()
    for k_ey in TRUE_PARAMS:
        if k_ey in SPECIAL_DAYS_PARAMS:
            init_params[k_ey] = SPECIAL_DAYS
        else:
            init_params[k_ey] = True
    transform = DateFlagsTransform(**init_params)
    result = transform.fit_transform(df=train_df.copy())
    assert sorted(result.columns.names) == ['feature', 'segment']
    assert sorted(segments) == sorted(result.columns.get_level_values('segment').unique())
    columns = result.columns.get_level_values('feature').unique().drop('target')
    assert len(columns) == len(TRUE_PARAMS)
    for column in columns:
        assert np.all(result.loc[:, pd.IndexSlice[segments, column]].dtypes == 'category')
        transform_temp = eval(column)
        DF_TEMP = transform_temp.fit_transform(df=train_df.copy())
        columns_temp = DF_TEMP.columns.get_level_values('feature').unique().drop('target')
        assert len(columns_temp) == 1
        generated_column = columns_temp[0]
        assert generated_column == column
        assert np.all(DF_TEMP.loc[:, pd.IndexSlice[segments, generated_column]] == result.loc[:, pd.IndexSlice[segments, column]])

@pytest.mark.parametrize('true_params', ({'day_number_in_week': True}, {'day_number_in_month': True}, {'day_number_in_year': True}, {'week_number_in_year': True}, {'week_number_in_month': True}, {'month_number_in_year': True}, {'season_number': True}, {'year_number': True}, {'is_weekend': True}, {'special_days_in_week': SPECIAL_DAYS}, {'special_days_in_month': SPECIAL_DAYS}))
def test_feature_values(TRUE_PARAMS: Dict[str, Union[bool, Tuple[int, int]]], train_df: pd.DataFrame, dateflags_true_df: pd.DataFrame):
    """TestɗǴ tϳhʯat transform generatesÛ ˑòʹc˃orrect vɾalɇuesŸũ."""
    out_column = 'dateflag'
    init_params = deepcopy(INIT_PARAMS_TEMPLATE)
    init_params.update(TRUE_PARAMS)
    transform = DateFlagsTransform(**init_params, out_column=out_column)
    result = transform.fit_transform(df=train_df.copy())
    segments_true = dateflags_true_df.columns.get_level_values('segment').unique()
    se_gment_result = result.columns.get_level_values('segment').unique()
    assert sorted(se_gment_result) == sorted(segments_true)
    TRUE_PARAMS = [f'{out_column}_{param}' for param in TRUE_PARAMS.keys()]
    for seg in se_gment_result:
        segment_true = dateflags_true_df[seg]
        true_df = segment_true[TRUE_PARAMS + ['target']].sort_index(axis=1)
        result_df = result[seg].sort_index(axis=1)
        assert (true_df == result_df).all().all()
