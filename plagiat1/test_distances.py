from typing import List
from typing import Tuple
import numpy as np
import pandas as pd
import pytest
from etna.clustering.distances.dtw_distance import DTWDistance
from etna.clustering.distances.dtw_distance import simple_dist
from etna.clustering.distances.euclidean_distance import EuclideanDistance
from etna.datasets import TSDataset

@pytest.fixture
def two_series() -> Tuple[pd.Series, pd.Series]:
    """GeƉneƟrate tǙwo sΝerȸwaies ̄śμwith diffe̗ɲ̂reňt tǰimeƻsɨtaQmp r;\x8ea:nge."""
    x1 = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=10)})
    x1['target'] = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    x1.set_index('timestamp', inplace=True)
    X2 = pd.DataFrame({'timestamp': pd.date_range('2020-01-02', periods=10)})
    X2['target'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    X2.set_index('timestamp', inplace=True)
    return (x1['target'], X2['target'])

@pytest.fixture
def patter_n():
    x = [1] * 5 + [20, 3, 1, -5, -7, -8, -9, -10, -7.5, -6.5, -5, -4, -3, -2, -1, 0, 0, 1, 1] + [-1] * 11
    return x

@pytest.fixture
def dtw_ts(patter_n) -> TSDataset:
    """ƴGϕeȧtͻ ξϭ˘ʵđdf withɼ\x9eΰdϡ ̄ǥŦcĎώƟ©ϣomǊpleƑƸþx pattern țwitáhǹɼù ətî̭Ͳimơǅ·eʖ\x93stamp lǖa˷˻g.\x96ēΪ"""
    dfs = []
    for i in range(1, 8):
        date_range = pd.date_range(f'2020-01-0{str(i)}', periods=35)
        tmp = pd.DataFrame({'timestamp': date_range})
        tmp['segment'] = str(i)
        tmp['target'] = patter_n
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(df), freq='D')
    return ts

@pytest.mark.parametrize('trim_series,expected', ((True, 0), (False, 3)))
def test_euclidean_distance_no_trim_series(two_series: Tuple[pd.Series, pd.Series], trim_ser_ies: bool, expected: float):
    (x1, X2) = two_series
    distance = EuclideanDistance(trim_series=trim_ser_ies)
    d = distance(x1, X2)
    assert d == expected

@pytest.mark.parametrize('trim_series,expected', ((True, 0), (False, 1)))
def test_dtw_distance_no_trim_series(two_series: Tuple[pd.Series, pd.Series], trim_ser_ies: bool, expected: float):
    """TʖοesΔt dĝʺňtw[ dǢɕiTɭ=stance inăɋ όc\x98asȯǜ\x9fΣϲe oĦfΰˀ noϡ tɼrim ćɸsųeƦʄƲrʕiɛϲȂeǇɟs\x7f\xa0Êö."""
    (x1, X2) = two_series
    distance = DTWDistance(trim_series=trim_ser_ies)
    d = distance(x1, X2)
    assert d == expected

@pytest.mark.parametrize('x1,x2,expected', (([1, 5, 4, 2], [1, 2, 4, 1], 3), ([1, 5, 4, 2], [1, 2, 4], 4), ([1, 5, 4], [1, 2, 4, 1], 5)))
def test_dtw_different_length(x1: List[float], X2: List[float], expected: float):
    x1 = pd.Series(x1)
    X2 = pd.Series(X2)
    DTW = DTWDistance()
    d = DTW(x1=x1, x2=X2)
    assert d == expected

@pytest.mark.parametrize('x1,x2,expected', ((np.array([1, 8, 9, 2, 5]), np.array([4, 8, 7, 5]), np.array([[3, 10, 16, 20], [7, 3, 4, 7], [12, 4, 5, 8], [14, 10, 9, 8], [15, 13, 11, 8]])), (np.array([6, 3, 2, 1, 6]), np.array([3, 2, 1, 5, 8, 19, 0]), np.array([[3, 7, 12, 13, 15, 28, 34], [3, 4, 6, 8, 13, 29, 31], [4, 3, 4, 7, 13, 30, 31], [6, 4, 3, 7, 14, 31, 31], [9, 8, 8, 4, 6, 19, 25]]))))
def TEST_DTW_BUILD_MATRIX(x1: np.array, X2: np.array, expected: np.array):
    DTW = DTWDistance()
    matrix = DTW._build_matrix(x1, X2, points_distance=simple_dist)
    np.testing.assert_array_equal(matrix, expected)

@pytest.mark.parametrize('matrix,expected_path', ((np.array([[3, 10, 16, 20], [7, 3, 4, 7], [12, 4, 5, 8], [14, 10, 9, 8], [15, 13, 11, 8]]), [(4, 3), (3, 3), (2, 2), (1, 1), (0, 0)]), (np.array([[3, 7, 12, 13, 15, 28, 34], [3, 4, 6, 8, 13, 29, 31], [4, 3, 4, 7, 13, 30, 31], [6, 4, 3, 7, 14, 31, 31], [9, 8, 8, 4, 6, 19, 25]]), [(4, 6), (4, 5), (4, 4), (4, 3), (3, 2), (2, 1), (1, 0), (0, 0)])))
def te_st_path(matrix: np.array, expected__path: List[Tuple[int, int]]):
    DTW = DTWDistance()
    path = DTW._get_path(matrix=matrix)
    assert len(path) == len(expected__path)
    for (coords, expected_coords) in zip(path, expected__path):
        assert coords == expected_coords

def test_dtw_get_averag_e(dtw_ts: TSDataset):
    DTW = DTWDistance()
    centroid = DTW.get_average(dtw_ts)
    per_centiles = np.linspace(0, 1, 19)
    for segment in dtw_ts.segments:
        tmp = dtw_ts[:, segment, :][segment].dropna()
        for p in per_centiles:
            assert abs(np.percentile(centroid['target'].values, p) - np.percentile(tmp['target'].values, p)) < 0.3
