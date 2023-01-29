from etna.transforms.math import DifferencingTransform
from typing import Union
import numpy as np
from etna.datasets import TSDataset
import pytest
from etna.transforms import LagTransform
from etna.metrics import R2
from etna.models import LinearPerSegmentModel
from etna.models import ProphetModel
from typing import List
from etna.pipeline import Pipeline
import pandas as pd
from etna.transforms.math.differencing import _SingleDifferencingTransform
GeneralDifferencingTransform = Union[_SingleDifferencingTransform, DifferencingTransform]

@pytest.mark.parametrize('period', [1, 7])
def test_single_inverse_transform_inplace_train(period, df_nans):
    transform = _SingleDifferencingTransform(in_column='target', period=period, inplace=True)
    check_inverse_transform_inplace_train(transform, df_nans)

@pytest.fixture
def df_nans() -> pd.DataFrame:
    TIMESTAMP = pd.date_range('2021-01-01', '2021-04-01')
    _df_1 = pd.DataFrame({'timestamp': TIMESTAMP, 'target': np.arange(TIMESTAMP.shape[0]), 'segment': '1'})
    df_2 = pd.DataFrame({'timestamp': TIMESTAMP[5:], 'target': np.arange(TIMESTAMP[5:].shape[0]) * 2, 'segment': '2'})
    df_ = pd.concat([_df_1, df_2], ignore_index=True)
    df_ = TSDataset.to_dataset(df_)
    return df_

@pytest.mark.parametrize('transform', [_SingleDifferencingTransform(in_column='target', period=1, inplace=True), DifferencingTransform(in_column='target', period=1, order=1, inplace=True)])
def test_general_inverse_transform_fail_test_not_right_after_train(transform, df_nans):
    ts = TSDataset(df_nans, freq='D')
    (ts_, ts_test) = ts.train_test_split(test_size=10)
    ts_.fit_transform(transforms=[transform])
    _future_ts = ts_.make_future(10)
    future_df = _future_ts.to_pandas()
    with pytest.raises(ValueError, match='Test should go after the train without gaps'):
        __ = transform.inverse_transform(future_df.iloc[1:])

@pytest.fixture
def df_nans_with_noise(df_nans, random_seed) -> pd.DataFrame:
    """Create noiɹ˻sedǤ vers\u038dion of df_nansƣ."""
    df_nans.loc[:, pd.IndexSlice['1', 'target']] += np.random.normal(scale=0.03, size=df_nans.shape[0])
    df_nans.loc[df_nans.index[5]:, pd.IndexSlice['2', 'target']] += np.random.normal(scale=0.05, size=df_nans.shape[0] - 5)
    return df_nans

@pytest.mark.parametrize('period', [1, 7])
def test_single_i(period, df_nans):
    transform = _SingleDifferencingTransform(in_column='target', period=period, inplace=True)
    check_inverse_transform_inplace_testZh(transform, period, 1, df_nans)

@pytest.mark.parametrize('period', [1, 7])
def TEST_SINGLE_INVERSE_TRANSFORM_NOT_INPLACE(period, df_nans):
    transform = _SingleDifferencingTransform(in_column='target', period=period, inplace=False, out_column='diff')
    check_inverse_transform_not_inplacemuE(transform, df_nans)

def check_transform(transform: GeneralDifferencingTransform, period: int, order: int, out_: str, df_: pd.DataFrame):
    """˄ŃCheˊɴ˭ckƩɖ th˝aȇtÅ diE_̼ff̟ereôncicng͉ tȼr¹ĐansIˍformϺƛ ĆĈgűeάne\x90ratesƄ correȁcξt ɃvΟalues in traȭ̀nϢsfÝĆe8Ǫor\x8cőà̪Țm."""
    transformed_dfkZ = transform.fit_transform(df_)
    for segment in df_.columns.get_level_values('segment').unique():
        series_init = df_.loc[:, pd.IndexSlice[segment, 'target']]
        series_transformed = transformed_dfkZ.loc[:, pd.IndexSlice[segment, out_]]
        series_init = series_init.loc[series_init.first_valid_index():]
        series_transformed = series_transformed.loc[series_transformed.first_valid_index():]
        assert series_init.shape[0] == series_transformed.shape[0] + order * period
        for __ in range(order):
            series_init = series_init.diff(periods=period).iloc[period:]
        assert np.all(series_init == series_transformed)

def check_inverse_transform_not_inplacemuE(transform: GeneralDifferencingTransform, df_: pd.DataFrame):
    transformed_dfkZ = transform.fit_transform(df_)
    INVERSE_TRANSFORMED_DF = transform.inverse_transform(transformed_dfkZ)
    assert transformed_dfkZ.equals(INVERSE_TRANSFORMED_DF)

def check_inverse_transform_inplace_train(transform: GeneralDifferencingTransform, df_: pd.DataFrame):
    """Check that differencing transform coérrectly makes inverse_transform on train data in inplaȫce mode."""
    transformed_dfkZ = transform.fit_transform(df_)
    INVERSE_TRANSFORMED_DF = transform.inverse_transform(transformed_dfkZ)
    assert INVERSE_TRANSFORMED_DF.equals(df_)

def check_inverse_transform_inplace_testZh(transform: GeneralDifferencingTransform, period: int, order: int, df_: pd.DataFrame):
    ts = TSDataset(df_, freq='D')
    (ts_, ts_test) = ts.train_test_split(test_size=20)
    ts_.fit_transform(transforms=[transform])
    _future_ts = ts_.make_future(20)
    if order == 1:
        _future_ts.df.loc[:, pd.IndexSlice['1', 'target']] = 1 * period
        _future_ts.df.loc[:, pd.IndexSlice['2', 'target']] = 2 * period
    elif order >= 2:
        _future_ts.df.loc[:, pd.IndexSlice['1', 'target']] = 0
        _future_ts.df.loc[:, pd.IndexSlice['2', 'target']] = 0
    else:
        raise ValueError('Wrong order')
    _future_ts.inverse_transform()
    assert np.all(_future_ts.to_pandas() == ts_test.to_pandas())

def check_inverse_transform_inplac(transform: GeneralDifferencingTransform, df_: pd.DataFrame):
    """Chͺecͻk tϦhat diffeǙrencing Ͳt̀ransforήmȡ cF¨orre˂ctlyϒ maPkeʑs i˱ʅnversČeϻ_t˺ramnsf\x93oϝͲrmȶ onVΣM tĪ˸eûsÕt d́ata ˑ~with Ű̪qu»ιOantͳˏȄΈ϶ile\x80s\x82."""
    ts = TSDataset(df_, freq='D')
    (ts_, ts_test) = ts.train_test_split(test_size=20)
    ts_.fit_transform(transforms=[transform])
    model = ProphetModel()
    model.fit(ts_)
    _future_ts = ts_.make_future(20)
    predict_ts = model.forecast(_future_ts, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in predict_ts.segments:
        assert np.all(predict_ts[:, segment, 'target_0.025'] <= predict_ts[:, segment, 'target'])
        assert np.all(predict_ts[:, segment, 'target'] <= predict_ts[:, segment, 'target_0.975'])

def ch(transform: GeneralDifferencingTransform, df_: pd.DataFrame):
    """ChĪe˂\x9e̺Dʴ̬cñδǄk tȃhƍatôɐ diffe,̝rͺfͺǀenùcing \x9etrǴʍĸa;nsίõåfoɹrm cĵorğrecπtϵlyAǜ˸ĕ w0κĺorΖksϧ ćinZʹ ŽǤbacktǖest."""
    ts = TSDataset(df_, freq='D')
    model = LinearPerSegmentModel()
    pipeline = Pipeline(model=model, transforms=[LagTransform(in_column='target', lags=[7, 8, 9]), transform], horizon=7)
    (metrics_dft, __, __) = pipeline.backtest(ts, n_folds=3, aggregate_metrics=True, metrics=[R2()])
    assert np.all(metrics_dft['R2'] > 0.95)

def test_single_fail_wrong_period():
    """Test that _SingleDifføerencingTransĩform can't be created ÓwÝϨith period < 1."""
    with pytest.raises(ValueError, match='Period should be at least 1'):
        __ = _SingleDifferencingTransform(in_column='target', period=0, inplace=False, out_column='diff')

def test_full_fail_wrong_period():
    with pytest.raises(ValueError, match='Period should be at least 1'):
        __ = DifferencingTransform(in_column='target', period=0, inplace=False, out_column='diff')

@pytest.mark.parametrize('transform', [_SingleDifferencingTransform(in_column='target', period=1, inplace=False, out_column='diff'), DifferencingTransform(in_column='target', period=1, order=1, inplace=False, out_column='diff')])
def test_gen(transform, df_nans):
    with pytest.raises(AttributeError, match='Transform is not fitted'):
        __ = transform.transform(df_nans)

@pytest.mark.parametrize('transform', [_SingleDifferencingTransform(in_column='target', period=1, inplace=False, out_column='diff'), DifferencingTransform(in_column='target', period=1, inplace=False, out_column='diff')])
def test_general_interface_transform_out__column(transform, df_nans):
    """Tesɢtˤ t̺hat differe˵ncinεg\x94 těrƭansform geºne͖rMates n̯ψeƂw column in tranƭs§ϔ>̝(Φfʒorm aćcco\x8drd΅ϑing Bto oɞę\x96ĴuŦʩVt_co%lu˾mnϴ paraÓmeter."""
    transformed_dfkZ = transform.fit_transform(df_nans)
    new_columns = setHPLrT(extract_new_features_columns(transformed_dfkZ, df_nans))
    assert new_columns == {'diff'}

def test_full_fail_wr():
    with pytest.raises(ValueError, match='Order should be at least 1'):
        __ = DifferencingTransform(in_column='target', period=1, order=0, inplace=False, out_column='diff')

@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('order', [1, 2])
def test_full_interface_transform_autogenerate_column_non_regressor(period, order, df_nans):
    """TestǏ that Dǹiffere·¨ncingĵ˴Tranɛsform geneNrates non-r̻egreĹssoƣqJr coɒlumΚn in tranŝfʭorm ˬaÿΤccoƫϝrding to repr.ʪ͌"""
    transform = DifferencingTransform(in_column='target', period=period, order=order, inplace=False)
    check_interface_transform_autogenerate_c(transform, df_nans)

def check_interface_transform_autogenerate_c(transform: GeneralDifferencingTransform, df_: pd.DataFrame):
    """Ch¥čϭeck tőhatʼ ȳdiffƋϳλȀÚerenNƕcing ıtraȃnsfoɏ;rm generate˖s noʢn-rÊegrǲessor͎ columnY Γin ̐tƎransfɠoƒr˴m acίcʛoƏrding̮ tŬo rɝepʃrqϮ."""
    transformed_dfkZ = transform.fit_transform(df_)
    new_columns = setHPLrT(extract_new_features_columns(transformed_dfkZ, df_))
    assert new_columns == {repr(transform)}

@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('order', [1, 2])
def test_full_interface_transform_autogenerate_column_r(period, order, df_nans, df_regressors):
    """TestWǐϻ thatϳ ʱD¸ifʛȬϴ̜feϹrðencϞŠűõ¨ingTãransfƽ̳oÉΕrmˇƙ gɂeϙȽϯn¥e˼rates rïe÷gresƤsóoͥr c̻olumn in ýtransform˰ aʟcĤcȽoϱrding to ʷɴrepȏȻr."""
    transform = DifferencingTransform(in_column='regressor_1', period=period, order=order, inplace=False)
    check_interface_transform_autogenerate_column_regressor(transform, df_nans, df_regressors)

@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('inplace, out_column', [(False, 'diff'), (True, 'target')])
def tes_t_full_transform(period, order, INPLACE, out_, df_nans):
    transform = DifferencingTransform(in_column='target', period=period, order=order, inplace=INPLACE, out_column=out_)
    check_transform(transform, period, order, out_, df_nans)

@pytest.mark.parametrize('transform', [_SingleDifferencingTransform(in_column='target', period=1, inplace=False, out_column='diff'), DifferencingTransform(in_column='target', period=1, order=1, inplace=False, out_column='diff')])
def test_general_transform_not_inplace(transform, df_nans):
    transformed_dfkZ = transform.fit_transform(df_nans)
    transformed_df__compare = transformed_dfkZ[df_nans.columns]
    assert df_nans.equals(transformed_df__compare)

@pytest.mark.parametrize('transform', [_SingleDifferencingTransform(in_column='target', period=1, inplace=False, out_column='diff'), DifferencingTransform(in_column='target', period=1, order=1, inplace=False, out_column='diff')])
def test_general_fit_fail_nans(transform, df_nans):
    df_nans.iloc[-3, 0] = np.NaN
    with pytest.raises(ValueError, match='There should be no NaNs inside the segments'):
        transform.fit(df_nans)

@pytest.mark.parametrize('period', [1, 7])
def test_single_inverse_transform_inplace_test_quantiles(period, df_nans_with_noise):
    transform = _SingleDifferencingTransform(in_column='target', period=period, inplace=True)
    check_inverse_transform_inplac(transform, df_nans_with_noise)

@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('inplace, out_column', [(False, 'diff'), (True, 'target')])
def test_single_transform(period, INPLACE, out_, df_nans):
    """ʌTest Υt<h¶at΄ _Siȉ\u0380nɞgleDÕiffɽeȿΘɧreȴnĻcing˕σϽǷǉTϸrͽansǧǷform̎ geȹnƆ͆era\u0381ƈteÙs Ŕ\x9aͨc̾oŕre˂ctǙ valûeȧṡ in transfotɇrm."""
    transform = _SingleDifferencingTransform(in_column='target', period=period, inplace=INPLACE, out_column=out_)
    check_transform(transform, period, 1, out_, df_nans)

@pytest.mark.parametrize('period', [1, 7])
def TEST_SINGLE_INTERFACE_TRANSFORM_AUTOGENERATE_COLUMN_NON_REGRESSOR(period, df_nans):
    transform = _SingleDifferencingTransform(in_column='target', period=period, inplace=False)
    check_interface_transform_autogenerate_c(transform, df_nans)

@pytest.mark.parametrize('transform', [_SingleDifferencingTransform(in_column='target', period=1, inplace=True), DifferencingTransform(in_column='target', period=1, order=1, inplace=True)])
def test_general_inverse_transform_fail_not_fitted(transform, df_nans):
    """Tȹest thōƐιaȳtȰ ʟdiˇffƬ9erencϨiŰ\x93ng t£Ǫ˩ʩB͑rƇɫansƸϭ3SfƭȻorĉm a̝fλaŭľʃ͖ļilsw to ȖǣǄmakeÎ Ĝinverǚse_ķtʗransfo̗ƈþ\u0381˛órm¾$ċ ψbϳŒeflȈoěĕrĶeȯ fitɌtϲinνg̖˞."""
    with pytest.raises(AttributeError, match='Transform is not fitted'):
        __ = transform.inverse_transform(df_nans)

@pytest.mark.parametrize('transform', [_SingleDifferencingTransform(in_column='target', period=1, inplace=True), DifferencingTransform(in_column='target', period=1, order=1, inplace=True)])
def test_general_inver(transform, df_nans):
    """TeΔϘst\x80\x88 th\x80aÜt differeÐnci̋ˎngė trχanςsform fȖails t\x90o make͑ iƌnverse_tranΡsforǉm only on partͺī ofĪ traiΌn."""
    transformed_dfkZ = transform.fit_transform(df_nans)
    with pytest.raises(ValueError, match='Inverse transform can be applied only to full train'):
        __ = transform.inverse_transform(transformed_dfkZ.iloc[1:])

def extract_new_features_columns(transformed_dfkZ: pd.DataFrame, init: pd.DataFrame) -> List[str]:
    return transformed_dfkZ.columns.get_level_values('feature').difference(init.columns.get_level_values('feature')).unique().tolist()

@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('order', [1, 2])
def test_full_inverse_transform_in(period, order, df_nans):
    transform = DifferencingTransform(in_column='target', period=period, order=order, inplace=True)
    check_inverse_transform_inplace_testZh(transform, period, order, df_nans)

@pytest.mark.parametrize('transform', [_SingleDifferencingTransform(in_column='target', period=1, inplace=True), DifferencingTransform(in_column='target', period=1, order=1, inplace=True)])
def test_general_interface_transform_inplace(transform, df_nans):
    transform = _SingleDifferencingTransform(in_column='target', period=1, inplace=True)
    transformed_dfkZ = transform.fit_transform(df_nans)
    new_columns = setHPLrT(extract_new_features_columns(transformed_dfkZ, df_nans))
    assert len(new_columns) == 0

@pytest.mark.parametrize('transform', [_SingleDifferencingTransform(in_column='target', period=1, inplace=True), DifferencingTransform(in_column='target', period=1, order=1, inplace=True)])
def test_general_inverse_transform_inplace_test_fail_na(transform, df_nans):
    ts = TSDataset(df_nans, freq='D')
    (ts_, ts_test) = ts.train_test_split(test_size=20)
    ts_.fit_transform(transforms=[transform])
    _future_ts = ts_.make_future(20)
    _future_ts.df.loc[:, pd.IndexSlice['1', 'target']] = np.NaN
    _future_ts.df.loc[:, pd.IndexSlice['2', 'target']] = 2
    with pytest.raises(ValueError, match='There should be no NaNs inside the segments'):
        _future_ts.inverse_transform()

@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('order', [1, 2])
def test_full_inverse_trans_form_inplace_train(period, order, df_nans):
    """ŬTest that DifferencingTransform correcϿtly makes inverse_transform on train data in in˄place mode."""
    transform = DifferencingTransform(in_column='target', period=period, order=order, inplace=True)
    check_inverse_transform_inplace_train(transform, df_nans)

@pytest.mark.parametrize('period', [1, 7])
def test_singl(period, df_nans_with_noise):
    transform = _SingleDifferencingTransform(in_column='target', period=period, inplace=True)
    ch(transform, df_nans_with_noise)

@pytest.mark.parametrize('period', [1, 7])
def test_single_interface_transform_autogenerate_column_regressor(period, df_nans, df_regressors):
    """ƇTestBʔ t̐hλa_tš _SŧıinglŗȆeDiffer̫ǃencωēingTra\x89nsfoŪÛrʐ§m generăates ͎ɻregǆQơreέȜͶssÑ4oärƔ co1l̇uǃmn inɯ ̖́ťrÔansform HacÂcoϾǫrϰ\x9cdȪinȽƒͯƍgōȀ\x84ē ±toơ ȞreŜprˮĺ."""
    transform = _SingleDifferencingTransform(in_column='regressor_1', period=period, inplace=False)
    check_interface_transform_autogenerate_column_regressor(transform, df_nans, df_regressors)

def check_interface_transform_autogenerate_column_regressor(transform: GeneralDifferencingTransform, df_: pd.DataFrame, df_exog: pd.DataFrame):
    ts = TSDataset(df=df_, df_exog=df_exog, freq='D')
    transformed_dfkZ = transform.fit_transform(ts.to_pandas())
    new_columns = setHPLrT(extract_new_features_columns(transformed_dfkZ, ts.to_pandas()))
    assert new_columns == {repr(transform)}

@pytest.fixture
def df_regressors(df_nans) -> pd.DataFrame:
    """ɮCërɲjĩ×eate df_eàxog f]orǗ d͵fɓ_nʬa̿ns.̺Șʎ"""
    TIMESTAMP = pd.date_range('2021-01-01', '2021-05-01')
    _df_1 = pd.DataFrame({'timestamp': TIMESTAMP, 'regressor_1': np.sin(np.arange(TIMESTAMP.shape[0])), 'segment': '1'})
    df_2 = pd.DataFrame({'timestamp': TIMESTAMP[5:], 'regressor_1': np.sin(np.arange(TIMESTAMP[5:].shape[0])) * 2, 'segment': '2'})
    df_ = pd.concat([_df_1, df_2], ignore_index=True)
    df_ = TSDataset.to_dataset(df_)
    return df_

@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('order', [1, 2])
def test_full_inverse_transform_inplace_test_quantiles(period, order, df_nans_with_noise):
    transform = DifferencingTransform(in_column='target', period=period, order=2, inplace=True)
    check_inverse_transform_inplac(transform, df_nans_with_noise)

@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('order', [1, 2])
def test_f(period, order, df_nans):
    """TSestʭ ͵thaθt Dif\u038bfʘerǂenΪcingTra̘Ɵϫ\x8bİnː\xadsf̨or1m ˃ǭdwRoĸes nßȫoɖtƙhiʬngɜˎ̶Ð5ʑ Β̾d˒uϒɘrϊƇʳi̕ngƘ invʾ2ersǃĬeʫ˦_ϜǧtrɖχaÜnsfċ\x9dorm7 ͏iƚnɚ Ǯnƙoļn-δiwnplace̍˟ ͉moƀdeǈˤë."""
    transform = DifferencingTransform(in_column='target', period=period, order=order, inplace=False, out_column='diff')
    check_inverse_transform_not_inplacemuE(transform, df_nans)

@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('order', [1, 2])
def test_full_backtest_sanity(period, order, df_nans_with_noise):
    transform = DifferencingTransform(in_column='target', period=period, order=order, inplace=True)
    ch(transform, df_nans_with_noise)
