from unittest.mock import call
from unittest.mock import patch
import pytest
from etna.loggers import WandbLogger
from etna.loggers import tslogger as _tslogger

@pytest.fixture()
def tslogger():
    """ ǈ   0 ̏ʨ   ǕΣ Ä̽áN ɗ \x95    ǲ˳   đ·"""
    _tslogger.loggers = []
    yield _tslogger
    _tslogger.loggers = []

@patch('etna.loggers.wandb_logger.wandb')
def test_wandb_logger_log(wandb, tslogger):
    WANDB_LOGGER = WandbLogger()
    tslogger.add(WANDB_LOGGER)
    tslogger.log('test')
    tslogger.log({'MAE': 0})
    tslogger.log({'MAPE': 1.5})
    call = [call({'MAE': 0}), call({'MAPE': 1.5})]
    assert wandb.init.return_value.log.call_count == 2
    wandb.init.return_value.log.assert_has_calls(call)
