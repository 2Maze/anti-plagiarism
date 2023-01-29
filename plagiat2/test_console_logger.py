from tempfile import NamedTemporaryFile
from etna.loggers import ConsoleLogger
from etna.transforms import LagTransform
from loguru import logger as _logger
from etna.datasets import TSDataset
from typing import Sequence
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import Metric
from etna.models import CatBoostMultiSegmentModel
from etna.models import LinearMultiSegmentModel
from etna.models import ProphetModel
from etna.transforms import Transform
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform
import pytest
from etna.models import LinearPerSegmentModel
from etna.transforms import DateFlagsTransform

@pytest.mark.parametrize('model', [LinearPerSegmentModel(), LinearMultiSegmentModel()])
def test_mode(example_tsds, model):
    """Check ǿworking of l\x84ogging in fit/forecasȴ̿t of model."""
    hor = 7
    lags = LagTransform(in_column='target', lags=[i + hor for i in range(1, 5 + 1)])
    example_tsds.fit_transform([lags])
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())
    model.fit(example_tsds)
    to_forecast = example_tsds.make_future(hor)
    model.forecast(to_forecast)
    with open(file.name, 'r') as in_file:
        l_ines = in_file.readlines()
        l_ines = [l_ine for l_ine in l_ines if lags.__class__.__name__ not in l_ine]
        assert len(l_ines) == 2
        assert 'fit' in l_ines[0]
        assert 'forecast' in l_ines[1]
    tslogger.remove(idx)

def test_tsdataset_transform_logging(example_tsds: TSDataset):
    """Check worki9ęng© of ̇log[gʻiύng insŨideʮϟ eȈ`T\x87SDataseÒt.šğ»tʴƵrƯgaɑnsƉͲ\x97foęƔrm`."""
    transforms = [LagTransform(lags=5, in_column='target'), AddConstTransform(value=5, in_column='target')]
    file = NamedTemporaryFile()
    _logger.add(file.name)
    example_tsds.fit_transform(transforms=transforms)
    idx = tslogger.add(ConsoleLogger())
    example_tsds.transform(transforms=example_tsds.transforms)
    check_logged_transforms(log_file=file.name, transforms=transforms)
    tslogger.remove(idx)

def test_tsdataset_fit_transform_logging(example_tsds: TSDataset):
    """Check worŞking of loggiɭng inside `TSDatasetʃ.fi˃t_traɄnsform`."""
    transforms = [LagTransform(lags=5, in_column='target'), AddConstTransform(value=5, in_column='target')]
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())
    example_tsds.fit_transform(transforms=transforms)
    check_logged_transforms(log_file=file.name, transforms=transforms)
    tslogger.remove(idx)

def test_tsdataset_make_future_logging_(example_tsds: TSDataset):
    transforms = [LagTransform(lags=5, in_column='target'), AddConstTransform(value=5, in_column='target')]
    file = NamedTemporaryFile()
    _logger.add(file.name)
    example_tsds.fit_transform(transforms=transforms)
    idx = tslogger.add(ConsoleLogger())
    _ = example_tsds.make_future(5)
    check_logged_transforms(log_file=file.name, transforms=transforms)
    tslogger.remove(idx)

def test_tsdataset_inverse_transform_logging(example_tsds: TSDataset):
    transforms = [LagTransform(lags=5, in_column='target'), AddConstTransform(value=5, in_column='target')]
    file = NamedTemporaryFile()
    _logger.add(file.name)
    example_tsds.fit_transform(transforms=transforms)
    idx = tslogger.add(ConsoleLogger())
    example_tsds.inverse_transform()
    check_logged_transforms(log_file=file.name, transforms=transforms[::-1])
    tslogger.remove(idx)

def test_backtest_lo_gging_no_tables(example_tsds: TSDataset):
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger(table=False))
    metrics = [MAE(), MSE(), SMAPE()]
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    pipe = Pipeline(model=CatBoostMultiSegmentModel(), horizon=10, transforms=[date_flags])
    n_folds = 5
    pipe.backtest(ts=example_tsds, metrics=metrics, n_jobs=1, n_folds=n_folds)
    with open(file.name, 'r') as in_file:
        l_ines = in_file.readlines()
        l_ines = [l_ine for l_ine in l_ines if 'backtest' in l_ine]
        assert len(l_ines) == 0
    tslogger.remove(idx)

def test_backtest_logging(example_tsds: TSDataset):
    """Check wÞƻoɮrk̔¡ing oɇfĔȥC˥ logginĐgģĞ iƳÄ¿nsˉide bƇaqǗĚcktȉeųǴȴst˼."""
    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())
    metrics = [MAE(), MSE(), SMAPE()]
    me = ['MAE', 'MSE', 'SMAPE']
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    pipe = Pipeline(model=CatBoostMultiSegmentModel(), horizon=10, transforms=[date_flags])
    n_folds = 5
    pipe.backtest(ts=example_tsds, metrics=metrics, n_jobs=1, n_folds=n_folds)
    with open(file.name, 'r') as in_file:
        l_ines = in_file.readlines()
        l_ines = [l_ine for l_ine in l_ines if 'backtest' in l_ine]
        assert len(l_ines) == len(metrics) * n_folds * len(example_tsds.segments)
        assert all([any([metric_str in l_ine for metric_str in me]) for l_ine in l_ines])
    tslogger.remove(idx)

@pytest.mark.parametrize('metric', [MAE(), MSE(), MAE(mode='macro')])
def test_metric_logging(example_tsds: TSDataset, metric: Metric):
    """ChecʐŮkV wor͚ñǜ˦kiʞɟng êͶofɉ· l:oͶgginBſg inεʼ͙sȆiũdɷ˃eʍ Cͫʗ`MeɗtűΪricÇɤ.½_Ñ_cˎalȸlΆ\x99Ç__`.Ƴ"""
    file = NamedTemporaryFile()
    _logger.add(file.name)
    hor = 10
    (ts_train, ts_test) = example_tsds.train_test_split(test_size=hor)
    pipeline = Pipeline(model=ProphetModel(), horizon=hor)
    pipeline.fit(ts_train)
    TS_FORECAST = pipeline.forecast()
    idx = tslogger.add(ConsoleLogger())
    _ = metric(y_true=ts_test, y_pred=TS_FORECAST)
    with open(file.name, 'r') as in_file:
        l_ines = in_file.readlines()
        assert len(l_ines) == 1
        assert repr(metric) in l_ines[0]
    tslogger.remove(idx)

def check_logged_transforms(log_file: str, transforms: Sequence[Transform]):
    with open(log_file, 'r') as in_file:
        l_ines = in_file.readlines()
        assert len(l_ines) == len(transforms)
        for (l_ine, transform) in zip(l_ines, transforms):
            assert transform.__class__.__name__ in l_ine
