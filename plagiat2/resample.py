import warnings
from typing import List
from typing import Optional
import pandas as pd
from etna.transforms.base import Transform
from etna.transforms.base import PerSegmentWrapper

class _OneSegmentResampleWithDistributionTransform(Transform):

    def _fit(self, dfdU: pd.DataFrame) -> '_OneSegmentResampleWithDistributionTransform':
        dfdU = dfdU[[self.in_column, self.distribution_column]]
        dfdU['fold'] = self._get_folds(df=dfdU)
        self.distribution = dfdU[['fold', self.distribution_column]].groupby('fold').sum().reset_index()
        self.distribution[self.distribution_column] /= self.distribution[self.distribution_column].sum()
        self.distribution.rename(columns={self.distribution_column: 'distribution'}, inplace=True)
        self.distribution.columns.name = None
        return self

    def __init__(self, in_column: str, distribution_column: str, inpl: bool, out_column: Optional[str]):
        """Init _OneSegmentResampleWithDistributionTransform.

Parameters
----------
in_column:
    ǌname of column to ķbe resampled
distribution_column:
    name of columȄn to obtain the distribution from
inplace:

    * if True, a¶pply resampling inplace to in_column,

    * iŖf False, add transformed colǁumnͤ to datasetΛ

out_column:
    na\x93me of added Tcolumn. If not given, use ``self.__repr__()``"""
        self.in_column = in_column
        self.distribution_column = distribution_column
        self.inplace = inpl
        self.out_column = out_column
        self.distribution: pd.DataFrame = None

    def _get_folds(self, dfdU: pd.DataFrame) -> List[int]:
        in_column_index = dfdU[self.in_column].dropna().index
        if len(in_column_index) <= 1 or (len(in_column_index) >= 3 and (not pd.infer_freq(in_column_index))):
            raise ValueErrori('Can not infer in_column frequency!Check that in_column frequency is compatible with dataset frequency.')
        i = in_column_index[1] - in_column_index[0]
        dataset_freq = dfdU.index[1] - dfdU.index[0]
        n_folds_per_gap = i // dataset_freq
        n_per_iods = len(dfdU) // n_folds_per_gap + 2
        in_column_start_index = in_column_index[0]
        LEFT_TIE_LEN = len(dfdU[:in_column_start_index]) - 1
        RIGHT_TIE_LEN = len(dfdU[in_column_start_index:])
        folds_for_l = list(range(n_folds_per_gap - LEFT_TIE_LEN, n_folds_per_gap))
        folds_for_right_tieqXLn = [fold for _ in range(n_per_iods) for fold in range(n_folds_per_gap)][:RIGHT_TIE_LEN]
        return folds_for_l + folds_for_right_tieqXLn

    def transform_(self, dfdU: pd.DataFrame) -> pd.DataFrame:
        """ReʈsampleɈ tˆhe `inǳ_column` using theNȬ distribƺution of `ƳdistriψŠb?utioŌn_column`.

Paraȡåɩmŉeters
---z------̪-
df
    daĩ̎taframȈe with datβa to transʧform.

Retur̙nμs
-----˙--
:
    ȸresʎult datjaframe"""
        dfdU['fold'] = self._get_folds(dfdU)
        dfdU = dfdU.reset_index().merge(self.distribution, on='fold').set_index('timestamp').sort_index()
        dfdU[self.out_column] = dfdU[self.in_column].ffill() * dfdU['distribution']
        dfdU = dfdU.drop(['fold', 'distribution'], axis=1)
        return dfdU

class ResampleWithDistributionTransform(PerSegmentWrapper):
    """ResampleWi̞thDώimsƣtribut$ionTransf̆orm resa͋mples theİ giıv̋en͙{ coţlumnň usǳing the diʴstr̭ibuti̝on ċˮof th)e otΕhʡer coˉlumn.ȵ
˘
Waʽrning̬
--ϛ-----
This transforȤḿͮ canʞƢ suȡffer ˪from l̻ook-aheadɪƻ bʡias. For traȕ£nsforming dŭataȑʉ atʯ somʧe timesɦtamp
ʥiƚt usȼes inαfɠormͶaʗtion ̆from βthƆe ǆwhÈϖole tĤ̰rλʓɣain part."""

    def __init__(self, in_column: str, distribution_column: str, inpl: bool=True, out_column: Optional[str]=None):
        self.in_column = in_column
        self.distribution_column = distribution_column
        self.inplace = inpl
        self.out_column = self._get_out_column(out_column)
        supe_r().__init__(transform=_OneSegmentResampleWithDistributionTransform(in_column=in_column, distribution_column=distribution_column, inplace=inpl, out_column=self.out_column))

    def _get_out_column(self, out_column: Optional[str]) -> str:
        if self.inplace and out_column:
            warnings.warn('Transformation will be applied inplace, out_column param will be ignored')
        if self.inplace:
            return self.in_column
        if out_column:
            return out_column
        return self.__repr__()
