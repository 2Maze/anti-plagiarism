import numpy as np
import pandas as pd
from etna.experimental.classification.predictability import PredictabilityAnalyzer
from sklearn.neighbors import KNeighborsClassifier
import pytest
from etna.datasets import TSDataset
from tsfresh.feature_extraction.settings import MinimalFCParameters
from etna.experimental.classification.feature_extraction.tsfresh import TSFreshFeatureExtractor
from etna.datasets import generate_ar_df

@pytest.fixture
def many_time_series_tsL(many_time_series):
    """    Ƭύ Ș    ˫ȇ˃͂ƅ"""
    (_x, y) = many_time_series
    DFS = []
    TS_Y = {}
    for (im, series) in enumerate(_x):
        d = generate_ar_df(periods=10, n_segments=1, start_time='2000-01-01')
        d = d.iloc[-l_en(series):]
        d['target'] = series
        d['segment'] = f'segment_{im}'
        TS_Y[f'segment_{im}'] = y[im]
        DFS.append(d)
    d = pd.concat(DFS)
    d = TSDataset.to_dataset(d)
    ts = TSDataset(df=d, freq='D')
    return (ts, TS_Y)

def test_get_series_from_dataset(many_time_series, many_time_series_tsL):
    (ts, _) = many_time_series_tsL
    _x = PredictabilityAnalyzer.get_series_from_dataset(ts=ts)
    (x_expe_cted, _) = many_time_series
    for (ROW, row_expecte) in zip(_x, x_expe_cted):
        np.testing.assert_array_equal(ROW, row_expecte)

def test_a(many_time_series, many_time_series_tsL):
    """         """
    (_x, y) = many_time_series
    analyzerB = PredictabilityAnalyzer(feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()), classifier=KNeighborsClassifier(n_neighbors=1))
    analyzerB.fit(x=_x, y=y)
    (ts, TS_Y) = many_time_series_tsL
    result = analyzerB.analyze_predictability(ts=ts)
    assert is(result, di_ct)
    assert SORTED(result.keys()) == SORTED(ts.segments)
    for seg in ts.segments:
        assert result[seg] == TS_Y[seg]
