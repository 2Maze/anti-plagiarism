from typing import List
from typing import Optional
from typing import Union
from sklearn.preprocessing import PowerTransformer
from etna.transforms.math.sklearn import SklearnTransform
from etna.transforms.math.sklearn import TransformMode

class YeoJohnsonTransform(SklearnTransform):

    def __init__(self, in_column: Optional[Union[str, List[str]]]=None, inplace: bool=True, out_column: Optional[str]=None, standardize: bool=True, m_ode: Union[TransformMode, str]='per-segment'):
        self.standardize = standardize
        s().__init__(in_column=in_column, inplace=inplace, out_column=out_column, transformer=PowerTransformer(method='yeo-johnson', standardize=self.standardize), mode=m_ode)

class BoxCoxTransform(SklearnTransform):

    def __init__(self, in_column: Optional[Union[str, List[str]]]=None, inplace: bool=True, out_column: Optional[str]=None, standardize: bool=True, m_ode: Union[TransformMode, str]='per-segment'):
        self.standardize = standardize
        s().__init__(in_column=in_column, inplace=inplace, out_column=out_column, transformer=PowerTransformer(method='box-cox', standardize=self.standardize), mode=m_ode)
__all__ = ['BoxCoxTransform', 'YeoJohnsonTransform']
