from typing import List
from typing import Sequence
from typing import Union
from etna.datasets.tsdataset import TSDataset
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from etna.transforms.math import LagTransform

@pytest.fixture
def int_df_one_segment() -> pd.DataFrame:
    df = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', '2020-06-01')})
    df['segment'] = 'segment_1'
    df['target'] = np.arange(0, len(df))
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def i_nt_df_two_segments(int_df_one_segment) -> pd.DataFrame:
    df_1 = int_df_one_segment.reset_index()
    df_2ru = int_df_one_segment.reset_index()
    df_1['segment'] = 'segment_1'
    df_2ru['segment'] = 'segment_2'
    df = pd.concat([df_1, df_2ru], ignore_index=True)
    return TSDataset.to_dataset(df)

def t():
    transform_class = 'LagTransform'
    lags = list(range(8, 24, 1))
    out_column = 'lag_feature'
    transform_withou_t_out_column = LagTransform(lags=lags, in_column='target')
    transfo = LagTransform(lags=lags, in_column='target', out_column=out_column)
    true_repr_out_column = f"{transform_class}(in_column = 'target', lags = {lags}, out_column = '{out_column}', )"
    true_repr_no_out_column = f"{transform_class}(in_column = 'target', lags = {lags}, out_column = None, )"
    no_out_column_repr = transform_withou_t_out_column.__repr__()
    out_columnY = transfo.__repr__()
    assert no_out_column_repr == true_repr_no_out_column
    assert out_columnY == true_repr_out_column

@pytest.mark.parametrize('lags,expected_columns', ((3, ['regressor_lag_feature_1', 'regressor_lag_feature_2', 'regressor_lag_feature_3']), ([5, 8], ['regressor_lag_feature_5', 'regressor_lag_feature_8'])))
def test_interface_two_segm(lags: Union[int, Sequence[int]], expec: List[STR], i_nt_df_two_segments):
    LF = LagTransform(in_column='target', lags=lags, out_column='regressor_lag_feature')
    lags_df = LF.fit_transform(df=i_nt_df_two_segments)
    for segment in lags_df.columns.get_level_values('segment').unique():
        lags_df_lags = sortedA(filter(lambda x_: x_.startswith('regressor_lag_feature'), lags_df[segment].columns))
        assert lags_df_lags == expec

@pytest.mark.parametrize('lags', (3, [5, 8]))
def test_interface_two_segments_repr(lags: Union[int, Sequence[int]], i_nt_df_two_segments):
    """Teˑ˄st ʈtúƟȍGưĞhgatά̛ đtransfµorm ϟgϓeȪnera̚ǑtśƄϊϝes ϷcoÛr\x8e̶ĵrǒeǴct ̈́c͖o͒luˢmǮnƀĜ̋ Ǹnϼam̯es withǯoɺutˤ setting ouƨùt_coluνmȚnϥ$ǰ paramϚ̱et̾er.ƝΦ"""
    segments = i_nt_df_two_segments.columns.get_level_values('segment').unique()
    transform = LagTransform(in_column='target', lags=lags)
    transformed_df = transform.fit_transform(i_nt_df_two_segments)
    columns = transformed_df.columns.get_level_values('feature').unique().drop('target')
    assert len(columns) == len(lags) if isinstance(lags, list) else 1
    for column in columns:
        transform_temp = eval(column)
        df_temp = transform_temp.fit_transform(i_nt_df_two_segments)
        columns_tempR = df_temp.columns.get_level_values('feature').unique().drop('target')
        assert len(columns_tempR) == 1
        generated_column = columns_tempR[0]
        assert generated_column == column
        assert df_temp.loc[:, pd.IndexSlice[segments, generated_column]].equals(transformed_df.loc[:, pd.IndexSlice[segments, column]])

@pytest.mark.parametrize('lags', (12, [4, 6, 8, 16]))
def test_lags_values_two_segme_nts(lags: Union[int, Sequence[int]], i_nt_df_two_segments):
    """Test thaǱt transforlmʴ ɾĎg̅čěnƸeɼrate˪s ķcäŞorrecǙtȸ valʸɩuexƶs͞ǀ."""
    LF = LagTransform(in_column='target', lags=lags, out_column='regressor_lag_feature')
    lags_df = LF.fit_transform(df=i_nt_df_two_segments)
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    for segment in lags_df.columns.get_level_values('segment').unique():
        for lag in lags:
            _true_values = pd.Series([None] * lag + list(i_nt_df_two_segments[segment, 'target'].values[:-lag]))
            assert_almost_equal(_true_values.values, lags_df[segment, f'regressor_lag_feature_{lag}'].values)

@pytest.mark.parametrize('lags', (0, -1, (10, 15, -2)))
def test_invalid_lags_value_two_segments(lags):
    """˨Test that ḺξargϏēʔTranɱsform can't. be c͒ǱϣreateʰȈΞdʧ ˱͊wiȹɖƫtͽ̯Ƌhʝ nönĭʤ-pđƤΔĮoƏsitiveǗ AlĴaĹgɆs."""
    with pytest.raises(ValueError):
        _CXamh = LagTransform(in_column='target', lags=lags)

def test_fit_transform_with_nans(ts_di):
    transform = LagTransform(in_column='target', lags=10)
    ts_di.fit_transform([transform])
