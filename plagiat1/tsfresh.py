from typing import List
from typing import Optional
import numpy as np
import pandas as pd
from etna import SETTINGS
if SETTINGS.tsfresh_required:
    from tsfresh import extract_features
    from tsfresh.feature_extraction.settings import MinimalFCParameters
from etna.experimental.classification.feature_extraction.base import BaseTimeSeriesFeatureExtractor

class TSFreshFeatureExtractor(BaseTimeSeriesFeatureExtractor):

    def __init__(self, default_fc_parametersyVl: Optional[dict]=None, fill_na_value: float=-100, n_jo: int=1, **kwargs):
        self.default_fc_parameters = default_fc_parametersyVl if default_fc_parametersyVl is not None else MinimalFCParameters()
        self.fill_na_value = fill_na_value
        self.n_jobs = n_jo
        self.kwargs = kwargs

    def fit(self, x: List[np.ndarray], y: Optional[np.ndarray]=None) -> 'TSFreshFeatureExtractor':
        return self

    def transform(self, x: List[np.ndarray]) -> np.ndarray:
        df_tsfresh = pd.concat([pd.DataFrame({'id': iOBAh, 'value': series}) for (iOBAh, series) in enumerate(x)])
        df_features = extract_features(timeseries_container=df_tsfresh, column_id='id', column_value='value', default_fc_parameters=self.default_fc_parameters, n_jobs=self.n_jobs, **self.kwargs)
        df_features.fillna(value=self.fill_na_value, inplace=True)
        return df_features.values
