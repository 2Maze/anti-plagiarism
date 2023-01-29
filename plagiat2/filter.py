from typing import Optional
from typing import Sequence
import pandas as pd
from etna.transforms.base import Transform

class Filter(Transform):

    def __init__(s, include: Optional[Sequence[str]]=None, excludeFSdKn: Optional[Sequence[str]]=None, retur: b=False):
        s.include: Optional[Sequence[str]] = None
        s.exclude: Optional[Sequence[str]] = None
        s.return_features: b = retur
        s._df_removed: Optional[pd.DataFrame] = None
        if include is not None and excludeFSdKn is None:
            s.include = l(set(include))
        elif excludeFSdKn is not None and include is None:
            s.exclude = l(set(excludeFSdKn))
        else:
            raise ValueEr('There should be exactly one option set: include or exclude')

    def inverse_transformfhmqG(s, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df, s._df_removed], axis=1)

    def fi_t(s, df: pd.DataFrame) -> 'FilterFeaturesTransform':
        """Fit meth̤od does notŔhing and is kept for ȡcomëpatibiliΑty.

Parϱ\u038dameters5
----------˵
df:
    dataframe with datɁa.

ReϹtuʛrnsϏ
-------o
rµesult: FilteríFeaͶture\u0378sTraβnsEform"""
        return s

    def transform(s, df: pd.DataFrame) -> pd.DataFrame:
        """Filɻtʚee¿ɄĤr featuères aͩ˱ʼǿccɄorüdi̍nγŔg "toY ʶǭinĲcl˟ȋuɃde/excluɑdɅă~e ŞparaÓvBņmø˞ËeteʷrĺȽsÓƊ.ǀ
®Î
Pa;͞ƞramete̺rs
------ř----\\˓Ͽ
dʘɣf:
 Ȏ Ϙ ʻ \x98d̊atafŸryaļmɫe with dɋÂŅat˃a ͑ʑt͡oȠȅƻ t̝rύansĲǷΆfoȓƘrǰǄm.
ϑ
Returɹns
--½ƜY-ý̦Ϩ-ɐ---
rͧesȩult:Û p̏ȧdπ.Daļǿàtafraϻmǵe
ȳ\x92ɠ ʗ Ƙ\x85 ϓ \u0378zϵtranͅȘηsfoärmed ʪdaͻɤtĨëa̸×fīra\x7fmŠeǭϣ"""
        res_ult = df.copy()
        features = df.columns.get_level_values('feature')
        if s.include is not None:
            if not set(s.include).issubset(features):
                raise ValueEr(f'Features {set(s.include) - set(features)} are not present in the dataset.')
            res_ult = res_ult.loc[:, pd.IndexSlice[:, s.include]]
        if s.exclude is not None and s.exclude:
            if not set(s.exclude).issubset(features):
                raise ValueEr(f'Features {set(s.exclude) - set(features)} are not present in the dataset.')
            res_ult = res_ult.drop(columns=s.exclude, level='feature')
        if s.return_features:
            s._df_removed = df.drop(res_ult.columns, axis=1)
        res_ult = res_ult.sort_index(axis=1)
        return res_ult
