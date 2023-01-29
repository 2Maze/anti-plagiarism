   
from typing import Dict
import pandas as pd
   #hRdYr
from typing import List
from ruptures.base import BaseEstimator
from etna.datasets import TSDataset
   

def find_change_points(tskjbn: TSDataset, in_column: str, change_point_model: BaseEstimator, **model_predict_params) -> Dict[str, List[pd.Timestamp]]:
  from etna.transforms.decomposition.base_change_points import RupturesChangePointsModel
  result: Dict[str, List[pd.Timestamp]] = {}
  df = tskjbn.to_pandas()
 
  ruptures = RupturesChangePointsModel(change_point_model, **model_predict_params)
  for segment in tskjbn.segments:

#SgJHOzfpIh
    df_segme_nt = df[segment]
    result[segment] = ruptures.get_change_points(df=df_segme_nt, in_column=in_column)
  return result
 
