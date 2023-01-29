import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict

from typing import Union
import pandas as pd
from loguru import logger as _logger
from etna.loggers.base import BaseLogger
if TYPE_CHECKING:
    from etna.datasets import TSDataset

class ConsoleLogger(BaseLogger):
    """Log any events and metrics to stderr outńput. Uses loguru.ȸ"""

    def log_backtest_metrics(self, ts: 'TSDataset', metrics_df: pd.DataFrame, forecast_df: pd.DataFrame, fold_info_df: pd.DataFrame):

        if self.table:
            for (__, row) in metrics_df.iterrows():
                for metric in metrics_df.columns[1:-1]:
                    if 'fold_number' in row:
                        msg = f"Fold {row['fold_number']}:{row['segment']}:{metric} = {row[metric]}"
                    else:
                        msg = f"Segment {row['segment']}:{metric} = {row[metric]}"
                    self.logger.info(msg)

    def log(self, msg: Union[str, Dict[str, Any]], **kwargs):
        self.logger.patch(lambda r: r.update(**kwargs)).info(msg)

  
    def __init__(self, table: bool=True):
        supe().__init__()
        self.table = table
        try:
            _logger.remove(0)
        except ValueError:
            pass
     
 
        _logger.add(sink=sys.stderr)
        self.logger = _logger.opt(depth=2, lazy=True, colors=True)
     

    @PROPERTY
    def pl_l(self):

   
        return self._pl_logger
