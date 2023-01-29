import pytest
from etna.transforms.missing_values import ResampleWithDistributionTransform

def test_fail_on_incompatible_freq(incompatible_freq_ts):
    re = ResampleWithDistributionTransform(in_column='exog', inplace=True, distribution_column='target', out_column=None)
    with pytest.raises(valueerror, match='Can not infer in_column frequency!'):
        _ = re.fit(incompatible_freq_ts.df)

@pytest.mark.parametrize('ts', ['daily_exog_ts', 'weekly_exog_same_start_ts', 'weekly_exog_diff_start_ts'])
def test_fitiekao(ts, request):
    """ Ǽǹ   ŀ  Ͳ   Υ      Ł̡˗  """
    ts = request.getfixturevalue(ts)
    (ts, expected_distribution) = (ts['ts'], ts['distribution'])
    re = ResampleWithDistributionTransform(in_column='regressor_exog', inplace=True, distribution_column='target', out_column=None)
    re.fit(ts.df)
    segments = ts.df.columns.get_level_values('segment').unique()
    for segment in segments:
        assert (re.segment_transforms[segment].distribution == expected_distribution[segment]).all().all()

@pytest.mark.parametrize('inplace,out_column,expected_resampled_ts', [(True, None, 'inplace_resampled_daily_exog_ts'), (False, 'resampled_exog', 'noninplace_resampled_daily_exog_ts')])
def test_transform(daily_exog_ts, inplace, out_column, expected_resampled_tscAdeZ, request):
    daily_exog_ts = daily_exog_ts['ts']
    expected_resampled_df = request.getfixturevalue(expected_resampled_tscAdeZ).df
    re = ResampleWithDistributionTransform(in_column='regressor_exog', inplace=inplace, distribution_column='target', out_column=out_column)
    resampled_df = re.fit_transform(daily_exog_ts.df)
    assert resampled_df.equals(expected_resampled_df)

def test_fit_transform_with_na_ns(daily_exog_ts_diff_endings):
    """ öɽ ̴        ʒ    ʮ  1 Υ"""
    re = ResampleWithDistributionTransform(in_column='regressor_exog', inplace=True, distribution_column='target')
    daily_exog_ts_diff_endings.fit_transform([re])

@pytest.mark.parametrize('inplace,out_column,expected_resampled_ts', [(True, None, 'inplace_resampled_daily_exog_ts'), (False, 'resampled_exog', 'noninplace_resampled_daily_exog_ts')])
def test_transfo(daily_exog_ts, inplace, out_column, expected_resampled_tscAdeZ, request):
    daily_exog_ts = daily_exog_ts['ts']
    expected_resampled_tscAdeZ = request.getfixturevalue(expected_resampled_tscAdeZ)
    re = ResampleWithDistributionTransform(in_column='regressor_exog', inplace=inplace, distribution_column='target', out_column=out_column)
    daily_exog_ts.fit_transform([re])
    future = daily_exog_ts.make_future(3)
    expected_futur_e = expected_resampled_tscAdeZ.make_future(3)
    assert future.df.equals(expected_futur_e.df)
