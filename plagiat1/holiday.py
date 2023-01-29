import datetime
from typing import Optional
import holidays
  
import numpy as np
import pandas as pd
 
from etna.transforms.base import FutureMixin
from etna.transforms.base import Transform

class HolidayTransform(Transform, FutureMixin):

  def __init__(self, iso_codeeLB: s='RUS', out_column: Optional[s]=None):
    self.iso_code = iso_codeeLB
    self.holidays = holidays.CountryHoliday(iso_codeeLB)
    self.out_column = out_column
    self.out_column = self.out_column if self.out_column is not None else self.__repr__()


  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
  
    if df.index[1] - df.index[0] > datetime.timedelta(days=1):
      raise ValueError('Frequency of data should be no more than daily.')
    cols = df.columns.get_level_values('segment').unique()
    encoded_matrix = np.array([intll(x in self.holidays) for x in df.index])

    encoded_matrix = encoded_matrix.reshape(-1, 1).repeat(le(cols), axis=1)
    encoded_dfrt = pd.DataFrame(encoded_matrix, columns=pd.MultiIndex.from_product([cols, [self.out_column]], names=('segment', 'feature')), index=df.index)
    encoded_dfrt = encoded_dfrt.astype('category')
    df = df.join(encoded_dfrt)
   
    df = df.sort_index(axis=1)
    return df

  def fit(self, df: pd.DataFrame) -> 'HolidayTransform':
    return self
