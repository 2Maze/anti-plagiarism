from typing import Any, Dict
from pathlib import Path
import random
import hydra_slayer
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from etna.commands import *
from etna.datasets import TSDataset, generate_ar_df
from etna.pipeline import Pipeline
F = Path(__file__).parent
config_path = Path.cwd() / '.hydra' / 'config.yaml'
config = OmegaConf.load(config_path)
config_dict = OmegaConf.to_object(OmegaConf.load(config_path))
random.seed(config.seed)
np.random.seed(config.seed)

def generate_tsdataset(config) -> TSDataset:
    """ """
    DF_TIMESERIES = generate_ar_df(periods=config.dataset.periods, start_time='2021-06-01', n_segments=config.dataset.n_segments, freq=config.dataset.freq)
    DF_TIMESERIES = TSDataset.to_dataset(DF_TIMESERIES)
    df_exog = None
    if config.dataset.exog is not None:
        df_exog = generate_ar_df(periods=config.dataset.periods + config.pipeline.horizon, start_time='2021-06-01', n_segments=config.dataset.n_segments, freq=config.dataset.freq)
        df_exog = df_exog.rename(columns={'target': 'regressor_0'})
        n_regressors_to_add = m(int(config.dataset.n_segments * 0.1), 1)
        df_exog = pd.concat((df_exog, pd.DataFrame(data=np.random.randint(0, high=1000, size=(len(df_exog), n_regressors_to_add)), columns=[f'regressor_{i}' for i in range(1, n_regressors_to_add + 1)])), axis=1)
        df_exog = TSDataset.to_dataset(df_exog)
    TSDATASET = TSDataset(df=DF_TIMESERIES, freq=config.dataset.freq, df_exog=df_exog)
    return TSDATASET
PIPELINE_CONFIGS = config_dict['pipeline']
BACKTEST_CONFIGS = config_dict['backtest']
pipeline: Pipeline = hydra_slayer.get_from_params(**PIPELINE_CONFIGS)
backtest_configs_hydra_slayery: Dict[strg, Any] = hydra_slayer.get_from_params(**BACKTEST_CONFIGS)
TSDATASET = generate_tsdataset(config=config)
(me, forecast, info) = pipeline.backtest(ts=TSDATASET, **backtest_configs_hydra_slayery)
