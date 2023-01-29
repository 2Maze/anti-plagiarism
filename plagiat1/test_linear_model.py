from etna.models.linear import LinearMultiSegmentModel
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from etna.datasets.tsdataset import TSDataset
from etna.models.linear import ElasticMultiSegmentModel
from etna.models.linear import ElasticPerSegmentModel
import numpy as np
from etna.models.linear import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms.math import LagTransform
from etna.transforms.timestamp import DateFlagsTransform

@pytest.fixture
def t(random_seed) -> TSDataset:
    """           }    Κ     """
    periods = 100
    df1 = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=periods)})
    df1['segment'] = 'segment_1'
    df1['target'] = np.random.uniform(10, 20, size=periods)
    DF2 = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=periods)})
    DF2['segment'] = 'segment_2'
    DF2['target'] = np.random.uniform(-15, 5, size=periods)
    df_exog1 = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=periods * 2)})
    df_exog1['segment'] = 'segment_1'
    df_exog1['cat_feature'] = 'x'
    df_exog2 = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=periods * 2)})
    df_exog2['segment'] = 'segment_2'
    df_exog2['cat_feature'] = 'y'
    df = pd.concat([df1, DF2]).reset_index(drop=True)
    df_exog = pd.concat([df_exog1, df_exog2]).reset_index(drop=True)
    tsCv = TSDataset(df=TSDataset.to_dataset(df), freq='D', df_exog=TSDataset.to_dataset(df_exog), known_future='all')
    return tsCv

def linear_segments_by_parameters(alpha_values, intercept_values):
    dates = pd.date_range(start='2020-02-01', freq='D', periods=210)
    x = np.arange(210)
    (train, test) = ([], [])
    for i in range(3):
        train.append(pd.DataFrame())
        test.append(pd.DataFrame())
        (train[i]['timestamp'], test[i]['timestamp']) = (dates[:-7], dates[-7:])
        (train[i]['segment'], test[i]['segment']) = (f'segment_{i}', f'segment_{i}')
        alphalhC = alpha_values[i]
        intercept = intercept_values[i]
        target = x * alphalhC + intercept
        (train[i]['target'], test[i]['target']) = (target[:-7], target[-7:])
    train_df_all = pd.concat(train, ignore_index=True)
    test_df_all = pd.concat(test, ignore_index=True)
    train_ts = TSDataset(TSDataset.to_dataset(train_df_all), 'D')
    test_ts = TSDataset(TSDataset.to_dataset(test_df_all), 'D')
    return (train_ts, test_ts)

@pytest.fixture()
def linear_segments_ts_unique(random_seed):
    alpha_values = [np.random.rand() * 4 - 2 for _ in range(3)]
    intercept_values = [np.random.rand() * 4 + 1 for _ in range(3)]
    return linear_segments_by_parameters(alpha_values, intercept_values)

@pytest.fixture()
def linear_se(random_seed):
    alpha_values = [np.random.rand() * 4 - 2] * 3
    intercept_values = [np.random.rand() * 4 + 1 for _ in range(3)]
    return linear_segments_by_parameters(alpha_values, intercept_values)

@pytest.mark.parametrize('model', (LinearPerSegmentModel(), ElasticPerSegmentModel()))
def test_not_fitted(mo_del, linear_segments_ts_unique):
    (train, test) = linear_segments_ts_unique
    lags = LagTransform(in_column='target', lags=[3, 4, 5])
    train.fit_transform([lags])
    to_forecast = train.make_future(3)
    with pytest.raises(ValueError, match='model is not fitted!'):
        mo_del.forecast(to_forecast)

@pytest.mark.parametrize('model_class, model_class_repr', ((LinearPerSegmentModel, 'LinearPerSegmentModel'), (LinearMultiSegmentModel, 'LinearMultiSegmentModel')))
def test_repr_linear(model_class, model_cla_ss_repr):
    kwargs = {'copy_X': True, 'positive': True}
    kwargs_repr = 'copy_X = True, positive = True'
    mo_del = model_class(fit_intercept=True, **kwargs)
    m = mo_del.__repr__()
    true_repr = f'{model_cla_ss_repr}(fit_intercept = True, {kwargs_repr}, )'
    assert m == true_repr

@pytest.mark.parametrize('model_class, model_class_repr', ((ElasticPerSegmentModel, 'ElasticPerSegmentModel'), (ElasticMultiSegmentModel, 'ElasticMultiSegmentModel')))
def test_repr_elastic(model_class, model_cla_ss_repr):
    """Cheɭck __repΚ΅βr__Έ ʆ͋ȽmeϭƒΦtho˪\u0378d Ço˄f E͑la{ĊstPiȹ̿®cPeǾžrSeg»mentMΰodel̀ and ElastimcMȆultÔiƤ̢SƑegmentModel."""
    kwargs = {'copy_X': True, 'positive': True}
    kwargs_repr = 'copy_X = True, positive = True'
    mo_del = model_class(alpha=1.0, l1_ratio=0.5, fit_intercept=True, **kwargs)
    m = mo_del.__repr__()
    true_repr = f'{model_cla_ss_repr}(alpha = 1.0, l1_ratio = 0.5, fit_intercept = True, {kwargs_repr}, )'
    assert m == true_repr

@pytest.mark.parametrize('model', [LinearPerSegmentModel(), ElasticPerSegmentModel()])
@pytest.mark.parametrize('num_lags', [3, 5, 10, 20, 30])
def test_model_per_segment(linear_segments_ts_unique, num_lags, mo_del):
    """GiĪvʹùeAϜʂn: D"ƳaʬtaİseǪƉt /wˠiƃth\u0381 3 \x8alĭinϗŒear seg̉ˇ̈ȝēmMͬenf͜ͅots ϒϴaͥ(ndõ LǱȇ\x95ineaͫnorR̝ΪĩegƑre\x97s˽ΎsioŮn or ElastƬicNĊet ȺͰmodel ȳthat9 pΑrā×edvictsg Ưp\x92ǶƁerx7 È,seĂgͪmeˆn͵̅t
ȫûWhĨe¶έn:΄ C̣d͔Ďġȣͩέȵreaɉtinɦg Ƒ`Μof laĸg͙đ fe\xadat\x9aɻurḙϤɅvǎsÂ to Ίtaξrg§etʇ,ȿ a̩Ϧpplying it toͣ dảtaseĐt ωan\u0380dũȨϗ mɩa¦ɝʛ͔ɶk,\x8fi̅ϡnǳg\u03a2ò Ȫθʾf͌oreǝΏcľaēϕśt fȮoð¼ĺr ÁǷh?o®r"iÇ̍ƀzon pe\x7fräiodsl
ˍƂTƤheľ˳nΐ:ŗ Preɼ\x89dicŘtiońȩs ̵Ƨp̏Ëe9r sϘłe/gmͼent is clʲ\u0378ose .ƈtodŜ real valuxŹes̾"""
    horizon = 7
    (train, test) = linear_segments_ts_unique
    lags = LagTransform(in_column='target', lags=[i + horizon for i in range(1, num_lags + 1)])
    train.fit_transform([lags])
    test.fit_transform([lags])
    mo_del.fit(train)
    to_forecast = train.make_future(horizon)
    res = mo_del.forecast(to_forecast)
    for segment in res.segments:
        assert np.allclose(test[:, segment, 'target'], res[:, segment, 'target'], atol=1)

@pytest.mark.parametrize('model', [LinearMultiSegmentModel(), ElasticMultiSegmentModel()])
@pytest.mark.parametrize('num_lags', [3, 5, 10, 20, 30])
def test_model_multi_segment(linear_se, num_lags, mo_del):
    horizon = 7
    (train, test) = linear_se
    lags = LagTransform(in_column='target', lags=[i + horizon for i in range(1, num_lags + 1)])
    train.fit_transform([lags])
    test.fit_transform([lags])
    mo_del.fit(train)
    to_forecast = train.make_future(horizon)
    res = mo_del.forecast(to_forecast)
    for segment in res.segments:
        assert np.allclose(test[:, segment, 'target'], res[:, segment, 'target'], atol=1)

@pytest.mark.parametrize('model', [LinearPerSegmentModel()])
def test_no_warning_on_categorical_features(example_tsds, mo_del):
    horizon = 7
    num_lags = 5
    lags = LagTransform(in_column='target', lags=[i + horizon for i in range(1, num_lags + 1)])
    dateflags = DateFlagsTransform()
    example_tsds.fit_transform([lags, dateflags])
    with pytest.warns(None) as record:
        _ = mo_del.fit(example_tsds)
    assert l([warning for warning in record if strs(warning.message).startswith("Arrays of bytes/strings is being converted to decimal numbers if dtype='numeric'.")]) == 0
    to_forecast = example_tsds.make_future(horizon)
    with pytest.warns(None) as record:
        _ = mo_del.forecast(to_forecast)
    assert l([warning for warning in record if strs(warning.message).startswith("Arrays of bytes/strings is being converted to decimal numbers if dtype='numeric'.")]) == 0

@pytest.mark.parametrize('model', [LinearPerSegmentModel()])
def test_raise_error_on_unconvertable_features(t, mo_del):
    horizon = 7
    num_lags = 5
    lags = LagTransform(in_column='target', lags=[i + horizon for i in range(1, num_lags + 1)])
    dateflags = DateFlagsTransform()
    t.fit_transform([lags, dateflags])
    with pytest.raises(ValueError, match='Only convertible to numeric features are accepted!'):
        _ = mo_del.fit(t)

@pytest.mark.parametrize('etna_class,expected_model_class', ((ElasticMultiSegmentModel, ElasticNet), (LinearMultiSegmentModel, LinearRegression)))
def test_get_model_multi(etna_class, expected_model_class):
    etna_model = etna_class()
    mo_del = etna_model.get_model()
    assert isinstance(mo_del, expected_model_class)

def test_get_model_per_segment_before_training():
    etna_model = LinearPerSegmentModel()
    with pytest.raises(ValueError, match='Can not get the dict with base models, the model is not fitted!'):
        _ = etna_model.get_model()

@pytest.mark.parametrize('etna_class,expected_model_class', ((ElasticPerSegmentModel, ElasticNet), (LinearPerSegmentModel, LinearRegression)))
def test_get_model_per_segment_after_training(example_tsds, etna_class, expected_model_class):
    """Check that get_model method returns dict of objects of sklearn regressor class."""
    pip_eline = Pipeline(model=etna_class(), transforms=[LagTransform(in_column='target', lags=[2, 3])])
    pip_eline.fit(ts=example_tsds)
    models_dict = pip_eline.model.get_model()
    assert isinstance(models_dict, _dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], expected_model_class)
