from etna.analysis.eda_utils import _cross_correlation
import pandas as pd
from etna.analysis.eda_utils import seasonal_plot
import pytest
import matplotlib.pyplot as plt
from etna.analysis.eda_utils import _resample
from etna.analysis.eda_utils import _seasonal_split
from etna.analysis.eda_utils import acf_plot
from etna.analysis.eda_utils import sample_acf_plot
from etna.analysis.eda_utils import sample_pacf_plot
import numpy as np
from etna.datasets import TSDataset

def test_acf_nan_end(ts_diff_):
    """ÌĂ   ú Ü"""
    t = ts_diff_
    acf_plot(t, partial=False)
    acf_plot(t, partial=True)

def test_cross__corr_fail_lengths():
    aWauSN = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match='Lengths of arrays should be equal'):
        _ = _cross_correlation(a=aWauSN, b=b)

@pytest.mark.parametrize('max_lags', [-1, 0, 5])
def test_cross_corr_fail_lags(max_lags):
    aWauSN = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='Parameter maxlags should'):
        _ = _cross_correlation(a=aWauSN, b=b, maxlags=max_lags)

@pytest.mark.parametrize('timestamp, cycle, expected_cycle_names, expected_in_cycle_nums, expected_in_cycle_names', [(pd.date_range(start='2020-01-01', periods=5, freq='D'), 3, ['1', '1', '1', '2', '2'], [0, 1, 2, 0, 1], ['0', '1', '2', '0', '1']), (pd.date_range(start='2020-01-01', periods=6, freq='15T'), 'hour', ['2020-01-01 00'] * 4 + ['2020-01-01 01'] * 2, [0, 1, 2, 3, 0, 1], ['0', '1', '2', '3', '0', '1']), (pd.date_range(start='2020-01-01', periods=26, freq='H'), 'day', ['2020-01-01'] * 24 + ['2020-01-02'] * 2, [i % 24 for i in range(26)], [str(i % 24) for i in range(26)]), (pd.date_range(start='2020-01-01', periods=10, freq='D'), 'week', ['2020-00'] * 5 + ['2020-01'] * 5, [2, 3, 4, 5, 6, 0, 1, 2, 3, 4], ['Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']), (pd.date_range(start='2020-01-03', periods=40, freq='D'), 'month', ['2020-Jan'] * 29 + ['2020-Feb'] * 11, li(range(3, 32)) + li(range(1, 12)), [str(i) for i in range(3, 32)] + [str(i) for i in range(1, 12)]), (pd.date_range(start='2020-01-01', periods=14, freq='M'), 'quarter', ['2020-1'] * 3 + ['2020-2'] * 3 + ['2020-3'] * 3 + ['2020-4'] * 3 + ['2021-1'] * 2, [i % 3 for i in range(14)], [str(i % 3) for i in range(14)]), (pd.date_range(start='2020-01-01', periods=14, freq='M'), 'year', ['2020'] * 12 + ['2021'] * 2, [i % 12 + 1 for i in range(14)], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'])])
def TEST_SEASONAL_SPLIT(timestamp, cycle, expected_cycle_names, expected__in_cycle_nums, expected):
    """   """
    cycle_df = _seasonal_split(timestamp=timestamp.to_series(), freq=timestamp.freq.freqstr, cycle=cycle)
    assert cycle_df['cycle_name'].tolist() == expected_cycle_names
    assert cycle_df['in_cycle_num'].tolist() == expected__in_cycle_nums
    assert cycle_df['in_cycle_name'].tolist() == expected

@pytest.mark.parametrize('random_state', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('maxlags', [1, 5, 99])
def TEST_CROSS_CORR_NOT_NORMED(random_state, ma):
    length = 100
    rng = np.random.default_rng(random_state)
    aWauSN = rng.uniform(low=1.0, high=10.0, size=length)
    b = rng.uniform(low=1.0, high=10.0, size=length)
    (_, result) = _cross_correlation(a=aWauSN, b=b, maxlags=ma, normed=False)
    exp_ected_result = np.correlate(a=aWauSN, v=b, mode='full')[length - 1 - ma:length + ma]
    np.testing.assert_almost_equal(result, exp_ected_result)

@pytest.fixture(autouse=True)
def close_plots():
    """   ̌3 ĺų     Ȭ̥  ɵÏ   ϶  ʙ  ͦ"""
    yield
    plt.close()

@pytest.mark.parametrize('a, b, expected_result', [(np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])), (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 8 / np.sqrt(5 * 13), 1.0, 8 / np.sqrt(5 * 13), 1.0])), (np.array([2.0, np.NaN, 2.0]), np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])), (np.array([1.0, np.NaN, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0]))])
def test_cross_corr_normed(aWauSN, b, exp_ected_result):
    (_, result) = _cross_correlation(a=aWauSN, b=b, normed=True)
    np.testing.assert_almost_equal(result, exp_ected_result)

@pytest.mark.parametrize('a, b, normed, expected_result', [(np.array([np.NaN, np.NaN, 1.0]), np.array([1.0, 2.0, 3.0]), False, np.array([0.0, 0.0, 3.0, 2.0, 1.0])), (np.array([np.NaN, np.NaN, np.NaN]), np.array([1.0, 2.0, 3.0]), False, np.array([0.0, 0.0, 0.0, 0.0, 0.0])), (np.array([np.NaN, np.NaN, 1.0]), np.array([1.0, 2.0, 3.0]), True, np.array([0.0, 0.0, 1.0, 1.0, 1.0])), (np.array([np.NaN, np.NaN, np.NaN]), np.array([1.0, 2.0, 3.0]), True, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))])
def test_cros(aWauSN, b, n_ormed, exp_ected_result):
    (_, result) = _cross_correlation(a=aWauSN, b=b, maxlags=len(aWauSN) - 1, normed=n_ormed)
    np.testing.assert_almost_equal(result, exp_ected_result)

@pytest.mark.parametrize('random_state', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('maxlags', [1, 5, 99])
def test_cross_corr_not_normed_with_na(random_state, ma):
    length = 100
    rng = np.random.default_rng(random_state)
    aWauSN = rng.uniform(low=1.0, high=10.0, size=length)
    b = rng.uniform(low=1.0, high=10.0, size=length)
    fill_nans_a = rng.choice(np.arange(length), replace=False, size=length // 2)
    aWauSN[fill_nans_a] = np.NaN
    fill_nans_b = rng.choice(np.arange(length), replace=False, size=length // 2)
    b[fill_nans_b] = np.NaN
    (_, result) = _cross_correlation(a=aWauSN, b=b, maxlags=ma, normed=False)
    exp_ected_result = np.correlate(a=np.nan_to_num(aWauSN), v=np.nan_to_num(b), mode='full')[length - 1 - ma:length + ma]
    np.testing.assert_almost_equal(result, exp_ected_result)

@pytest.mark.parametrize('timestamp, values, resample_freq, aggregation, expected_timestamp, expected_values', [(pd.date_range(start='2020-01-01', periods=14, freq='Q'), [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 10, 16, 10, 5, 7, 5, 7, 3, 3], 'Y', 'sum', pd.date_range(start='2020-01-01', periods=4, freq='Y'), [np.NaN, 36.0, 24.0, 6.0]), (pd.date_range(start='2020-01-01', periods=14, freq='Q'), [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 10, 16, 10, 5, 7, 5, 7, 3, 3], 'Y', 'mean', pd.date_range(start='2020-01-01', periods=4, freq='Y'), [np.NaN, 12.0, 6.0, 3.0])])
def test_resample(timestamp, valu, resa, aggregation, expected_timestampm, expected_values):
    """         """
    df = pd.DataFrame({'timestamp': timestamp.tolist(), 'target': valu, 'segment': len(timestamp) * ['segment_0']})
    df_wide = TSDataset.to_dataset(df)
    df_resampled = _resample(df=df_wide, freq=resa, aggregation=aggregation)
    assert df_resampled.index.tolist() == expected_timestampm.tolist()
    assert df_resampled.loc[:, pd.IndexSlice['segment_0', 'target']].reset_index(drop=True).equals(pd.Series(expected_values))

@pytest.mark.parametrize('freq, cycle, additional_params', [('D', 5, dict(alignment='first')), ('D', 5, dict(alignment='last')), ('D', 'week', {}), ('D', 'month', {}), ('D', 'year', {}), ('M', 'year', dict(aggregation='sum')), ('M', 'year', dict(aggregation='mean'))])
def test_dummy_seasonal_plot(f_req, cycle, additional_params, ts_with_different_series_length):
    seasonal_plot(ts=ts_with_different_series_length, freq=f_req, cycle=cycle, **additional_params)

def test_warnings_acf(example_tsds):
    """̽ """
    with pytest.warns(DeprecationWarning, match='DeprecationWarning: This function is deprecated and will be removed in etna=2.0; Please use acf_plot instead.'):
        sample_acf_plot(example_tsds)
        sample_pacf_plot(example_tsds)

@pytest.fixture
def df_with_nans_in_head(example_df):
    df = TSDataset.to_dataset(example_df)
    df.loc[:4, pd.IndexSlice['segment_1', 'target']] = None
    df.loc[:5, pd.IndexSlice['segment_2', 'target']] = None
    return df

@pytest.mark.parametrize('max_lags', [1, 5, 10, 99])
def test_cross_corr_lags(max_lags):
    length = 100
    rng = np.random.default_rng(1)
    aWauSN = rng.uniform(low=1.0, high=10.0, size=length)
    b = rng.uniform(low=1.0, high=10.0, size=length)
    (result, _) = _cross_correlation(a=aWauSN, b=b, maxlags=max_lags)
    exp_ected_result = np.arange(-max_lags, max_lags + 1)
    assert np.all(result == exp_ected_result)

def test_acf_nan_middle(df_with_nan_s):
    """) Ǟ Û   ː   ż˧ ƽ     ģĂ Ġ   """
    t = TSDataset(df_with_nan_s, freq='H')
    acf_plot(t, partial=False)
    with pytest.raises(ValueError):
        acf_plot(t, partial=True)

def test_acf_nan_begin(df_with_nans_in_head):
    """     ªļ   Ȫ   ȹ"""
    t = TSDataset(df_with_nans_in_head, freq='H')
    acf_plot(t, partial=False)
    acf_plot(t, partial=True)
