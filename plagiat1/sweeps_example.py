import random
from typing import Optional
import hydra
import hydra_slayer
from etna.pipeline import Pipeline
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pathlib import Path
from etna.datasets import TSDataset
from etna.loggers import WandbLogger
from etna.loggers import tslogger
import numpy as np
OmegaConf.register_new_resolver('range', lambda x, y: list(range(x, y)))
OmegaConf.register_new_resolver('sum', lambda x, y: x + y)
FILE_PATH = Path(__file__)

def set_seed(seed: int=42):
    """į """
    random.seed(seed)
    np.random.seed(seed)

def init_logger(config: dict, project: str='wandb-sweeps', tags: Optional[list]=['test', 'sweeps']):
    tslogger.loggers = []
    wblogger = WandbLogger(project=project, tags=tags, config=config)
    tslogger.add(wblogger)

def dataloader(file_path: Path, freq: str) -> TSDataset:
    df = pd.read_csv(file_path)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=freq)
    return ts

@hydra.main(config_name='config.yaml')
def objec(cfg: DictConfig):
    """           Ŧ      """
    config = OmegaConf.to_container(cfg, resolve=True)
    set_seed(cfg.seed)
    ts = dataloader(file_path=cfg.dataset.file_path, freq=cfg.dataset.freq)
    pipeline: Pipeline = hydra_slayer.get_from_params(**config['pipeline'])
    backtest_params = hydra_slayer.get_from_params(**config['backtest'])
    init_logger(pipeline.to_dict())
    (_, _, _) = pipeline.backtest(ts, **backtest_params)
if __name__ == '__main__':
    objec()
