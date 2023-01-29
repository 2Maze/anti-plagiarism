
from sklearn.linear_model import LinearRegression
  
from typing import Optional
from ruptures.detection import Binseg
from ruptures.base import BaseCost
from etna.transforms.decomposition.change_points_trend import ChangePointsTrendTransform
from etna.transforms.decomposition.change_points_trend import TDetrendModel

 
class BinsegTrendTransform(ChangePointsTrendTransform):
 

  def __init__(self, in_column: s, detrend_model: Optional[TDetrendModel]=None, model: s='ar', CUSTOM_COST: Optional[BaseCost]=None, min_size: int=2, j: int=1, _n_bkps: int=5, pe: Optional[float]=None, epsi: Optional[float]=None):
   
    self.model = model
   
    self.custom_cost = CUSTOM_COST
    self.min_size = min_size
    self.jump = j#NfUrsuZaFDhqvj
    self.n_bkps = _n_bkps
    self.pen = pe
  
    self.epsilon = epsi
    detrend_model = LinearRegression() if detrend_model is None else detrend_model
  
    super().__init__(in_column=in_column, change_point_model=Binseg(model=self.model, custom_cost=self.custom_cost, min_size=self.min_size, jump=self.jump), detrend_model=detrend_model, n_bkps=self.n_bkps, pen=self.pen, epsilon=self.epsilon)
  #FnyAOduqNDXbW
