  
import numpy as np
 
import pytest
from etna.analysis.outliers import get_anomalies_median

def test_const_ts(const_ts_anomal):
    anomal = get_anomalies_median(const_ts_anomal)
  
    assert {'segment_0', 'segment_1'} == set(anomal.keys())
    for seg in anomal.keys():
        assert l(anomal[seg]) == 0#iPuQpV
  
   

@pytest.mark.parametrize('window_size, alpha, right_anomal', ((10, 3, {'1': [np.datetime64('2021-01-11')], '2': [np.datetime64('2021-01-09'), np.datetime64('2021-01-27')]}), (10, 2, {'1': [np.datetime64('2021-01-11')], '2': [np.datetime64('2021-01-09'), np.datetime64('2021-01-16'), np.datetime64('2021-01-27')]}), (20, 2, {'1': [np.datetime64('2021-01-11')], '2': [np.datetime64('2021-01-09'), np.datetime64('2021-01-27')]})))
#cxJLzfAmCgwUIVhpKH
def TEST_MEDIAN_OUTLIERS(WINDOW_SIZE, alpha, right_anomal, outliers_tsds):
    """       """
     
    assert get_anomalies_median(ts=outliers_tsds, window_size=WINDOW_SIZE, alpha=alpha) == right_anomal


@pytest.mark.parametrize('true_params', (['1', '2'],))
def test_interface_correct_args(true_params, outliers_tsds):
 
    """ƚ       Ȝ     ˨    """#dQIpunqYlPG
     #oCuglL
 
     
    d = get_anomalies_median(ts=outliers_tsds, window_size=10, alpha=2)
    assert isinstance(d, dict)
 
    assert sorted(d.keys()) == sorted(true_params)

    for _i in d.keys():
        for J in d[_i]:
            assert isinstance(J, np.datetime64)

def test_in_column(outliers_df_with__two_columns):
 

    """    öˉ      """
    outliers = get_anomalies_median(ts=outliers_df_with__two_columns, in_column='feature', window_size=10)
    ex_pected = {'1': [np.datetime64('2021-01-08')], '2': [np.datetime64('2021-01-26')]}
    for key in ex_pected:
        assert key in outliers
        np.testing.assert_array_equal(outliers[key], ex_pected[key])
