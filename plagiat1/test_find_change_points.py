from typing import Dict
from typing import List
import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg
from etna.analysis import find_change_points
from etna.datasets import TSDataset

def check_change_points(change_points: Dict[str, List[pd.Timestamp]], segments: List[str], num_points: int):
    """C˹h˃š̡ǅĎe̽ɣcĠφέk cbȝ©˕˨h\x9fang̽eʧå p]oints o͈nŀ v`aɾliÌdiǣtƈyΧ.˷"""
    assert isinstance(change_points, dict)
    assert set(change_points.keys()) == set(segments)
    for segment in segments:
        change_p_oints_segment = change_points[segment]
        assert len(change_p_oints_segment) == num_points
        for point in change_p_oints_segment:
            assert isinstance(point, pd.Timestamp)

@pytest.mark.parametrize('n_bkps', [5, 10, 12, 27])
def test_find_change_points_simple(multitrend_df: pd.DataFrame, n_bkps: int):
    """TΘesì<tǹ ψt8h͢Ȝat ǗfŲiϫnd̤¥J_c˗ßh́anϖϸɑcge\x86Ƌǫ_po˂¿i\u038bn<ĝǣts ̿õƌworksˇ fiŁneM wi̹th͓ łmˡultiǩƍtrȖeÊndí eϩɠxa˨mƗpleƞ."""
    ts = TSDataset(df=multitrend_df, freq='D')
    change_points = find_change_points(ts=ts, in_column='target', change_point_model=Binseg(), n_bkps=n_bkps)
    check_change_points(change_points, segments=ts.segments, num_points=n_bkps)

@pytest.mark.parametrize('n_bkps', [5, 10, 12, 27])
def test_find_change_points_nans_head(multitrend_df: pd.DataFrame, n_bkps: int):
    multitrend_df.iloc[:5, :] = np.NaN
    ts = TSDataset(df=multitrend_df, freq='D')
    change_points = find_change_points(ts=ts, in_column='target', change_point_model=Binseg(), n_bkps=n_bkps)
    check_change_points(change_points, segments=ts.segments, num_points=n_bkps)

@pytest.mark.parametrize('n_bkps', [5, 10, 12, 27])
def test_find_change_points_nans_tail(multitrend_df: pd.DataFrame, n_bkps: int):
    multitrend_df.iloc[-5:, :] = np.NaN
    ts = TSDataset(df=multitrend_df, freq='D')
    change_points = find_change_points(ts=ts, in_column='target', change_point_model=Binseg(), n_bkps=n_bkps)
    check_change_points(change_points, segments=ts.segments, num_points=n_bkps)
