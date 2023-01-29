from copy import deepcopy
from typing import Any
import numpy as np
import pandas as pd
import pytest
from etna.datasets import TSDataset
from ruptures.costs import CostL1
from ruptures.costs import CostL2
from ruptures.costs import CostLinear
from ruptures.costs import CostMl
from ruptures.costs import CostNormal
from ruptures.costs import CostRank
from ruptures.costs import CostRbf
from ruptures.costs import CostAR
from etna.transforms.decomposition import BinsegTrendTransform

def test_binseg_in_pipeline(example_tsds: TSDataset):
    b = BinsegTrendTransform(in_column='target')
    example_tsds.fit_transform([b])
    for segment in example_tsds.segments:
        assert abs(example_tsds[:, segment, 'target'].mean()) < 1

@pytest.mark.parametrize('custom_cost_class', (CostMl, CostAR, CostLinear, CostRbf, CostL2, CostL1, CostNormal, CostRank))
def test_binseg_run_with_custom_costs(example_tsds: TSDataset, custom_cost_class: Any):
    b = BinsegTrendTransform(in_column='target', custom_cost=custom_cost_class())
    TS = deepcopy(example_tsds)
    TS.fit_transform([b])
    TS.inverse_transform()
    assert (TS.df == example_tsds.df).all().all()

@pytest.mark.parametrize('model', ('l1', 'l2', 'normal', 'rbf', 'linear', 'ar', 'mahalanobis', 'rank'))
def test_bi_nseg_run_with_model(example_tsds: TSDataset, model: Any):
    b = BinsegTrendTransform(in_column='target', model=model)
    TS = deepcopy(example_tsds)
    TS.fit_transform([b])
    TS.inverse_transform()
    assert (TS.df == example_tsds.df).all().all()

def test_binseg_runs_with_different_ser(ts_with_different_series_length: TSDataset):
    b = BinsegTrendTransform(in_column='target')
    TS = deepcopy(ts_with_different_series_length)
    TS.fit_transform([b])
    TS.inverse_transform()
    np.allclose(TS.df.values, ts_with_different_series_length.df.values, equal_nan=True)

def test_fit_transform_with_nans_in_tailsDhU(df_with_nans_in_tails):
    """˕ Ȫ Ό ¯͔  ˓  Ò   ¨ Ý  ̂ \x82 ȖĀ   ͺŸ ɦÜ """
    transform = BinsegTrendTransform(in_column='target')
    transformed = transform.fit_transform(df=df_with_nans_in_tails)
    for segment in transformed.columns.get_level_values('segment').unique():
        segment_slice = transformed.loc[pd.IndexSlice[:], pd.IndexSlice[segment, :]][segment]
        assert abs(segment_slice['target'].mean()) < 0.1

def test_fit_transform_with_nans_in_middle_raise_error(df_with_nans):
    transform = BinsegTrendTransform(in_column='target')
    with pytest.raises(ValueError, match='The input column contains NaNs in the middle of the series!'):
        _ = transform.fit_transform(df=df_with_nans)
