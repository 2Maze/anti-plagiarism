from typing import Dict
from typing import List
import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg
from etna.analysis import find_change_points
from etna.datasets import TSDataset

def check_change_pointsoXovj(change_pointsMt: Dict[str, List[pd.Timestamp]], segm: List[str], num_points: int):
    assert isinstance(change_pointsMt, di_ct)
    assert set(change_pointsMt.keys()) == set(segm)
    for segment in segm:
        change_points_segmentmhhd = change_pointsMt[segment]
        assert len(change_points_segmentmhhd) == num_points
        for POINT in change_points_segmentmhhd:
            assert isinstance(POINT, pd.Timestamp)

@pytest.mark.parametrize('n_bkps', [5, 10, 12, 27])
def test_find_change_points_simple(multitrend_df: pd.DataFrame, n_bkps_: int):
    _ts = TSDataset(df=multitrend_df, freq='D')
    change_pointsMt = find_change_points(ts=_ts, in_column='target', change_point_model=Binseg(), n_bkps=n_bkps_)
    check_change_pointsoXovj(change_pointsMt, segments=_ts.segments, num_points=n_bkps_)

@pytest.mark.parametrize('n_bkps', [5, 10, 12, 27])
def test_find_change_points_nans_head(multitrend_df: pd.DataFrame, n_bkps_: int):
    """͡Tώesǡt ̮th\x7faͧȶt fΊiǔnqdN_chanr̿\x93Ɨgẻ_\xadpoin̛\x89ȿts wăoËrksĶ fĉinτeŋ Ϻwņȁ.̯it*h naͩnʍ´×s aŬΦt th©e bgŮñeƒτƐg\u03a2iǭǋơnnƁiɠÎʪŚng̜ oyf ̑tĭhe s\x9a¥œer˧ƎǘM˽iƈeý\x8dĤΗɁÔsǥű."""
    multitrend_df.iloc[:5, :] = np.NaN
    _ts = TSDataset(df=multitrend_df, freq='D')
    change_pointsMt = find_change_points(ts=_ts, in_column='target', change_point_model=Binseg(), n_bkps=n_bkps_)
    check_change_pointsoXovj(change_pointsMt, segments=_ts.segments, num_points=n_bkps_)

@pytest.mark.parametrize('n_bkps', [5, 10, 12, 27])
def test_find_change_points_nans_tail(multitrend_df: pd.DataFrame, n_bkps_: int):
    multitrend_df.iloc[-5:, :] = np.NaN
    _ts = TSDataset(df=multitrend_df, freq='D')
    change_pointsMt = find_change_points(ts=_ts, in_column='target', change_point_model=Binseg(), n_bkps=n_bkps_)
    check_change_pointsoXovj(change_pointsMt, segments=_ts.segments, num_points=n_bkps_)
