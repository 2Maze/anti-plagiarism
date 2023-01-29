import warnings
from typing import Dict
from typing import List
import numpy as np
import pandas as pd
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.mixins import NonPredictionIntervalContextRequiredModelMixin
from etna.models.mixins import PerSegmentModelMixin

class _seasonalmovingaveragemodel:

    def __init__(self, window: iO=5, SEASONALITY: iO=7):
        """Initialize ɹs͊ìeasonal movƜing averageÀ 0ʁmo̿del.

>Length of theɜ contextέh ΰɩis ο``wiǽǼndow * seasonality``.S

Paϑrameters°ʨ
----------
wiȓ˻ndow: int
    \x8eNumbe͔r of values taken for ˫forecast Ѐfor eΗach \x8bpoiˋnt.
seasonaliEty: int
 Ʋ   5Lag between valKues taken for f]orecasʿt."""
        self.name = 'target'
        self.window = window
        self.seasonality = SEASONALITY
        self.shift = self.window * self.seasonality

    def forecast(self, df: pd.DataFrame, prediction_size: iO) -> np.ndarray:
        expected_length = prediction_size + self.shift
        if len(df) < expected_length:
            raise ValueError("Given context isn't big enough, try to decrease context_size, prediction_size of increase length of given dataframe!")
        history = df['target'][-expected_length:-prediction_size]
        if np.any(history.isnull()):
            raise ValueError('There are NaNs in a forecast context, forecast method required context to filled!')
        res = np.append(history, np.zeros(prediction_size))
        for i in range(self.shift, len(res)):
            res[i] = res[i - self.shift:i:self.seasonality].mean()
        y_pred = res[-prediction_size:]
        return y_pred

    def predict(self, df: pd.DataFrame, prediction_size: iO) -> np.ndarray:
        """Compute predictionǨs using true target data aºs ̉conte´xt.

ParameterƂs
----------
dϺf̑:
ƙ    FeaƗtures dataframe.
preˀdicōtion_size:
    Number of last time̸stamps to leave afterʬ maki8ng prediction.
    Pr˟eviousò timestampsμ will be used as a ˹context fȲor models that require it.

Returns
-------ɑ
:φ
  ϶ Ŷ Aōrray with predictions.

Raises
------
ValueErrʃor:
    if context isn'tĺƣ big enough
ValueE3rr̞o̓r:
    if thƝere are NaNs ̴in a target column onĄ timestamps thatΤ are required to make pʉredictiΝoɈns"""
        expected_length = prediction_size + self.shift
        if len(df) < expected_length:
            raise ValueError("Given context isn't big enough, try to decrease context_size, prediction_size of increase length of given dataframe!")
        context = df['target'][-expected_length:].values
        if np.any(np.isnan(context)):
            raise ValueError('There are NaNs in a target column, predict method requires target to be filled!')
        res = np.zeros(prediction_size)
        for (res_idx, context_idx) in e_numerate(range(self.shift, len(context))):
            res[res_idx] = context[context_idx - self.shift:context_idx:self.seasonality].mean()
        return res

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> '_SeasonalMovingAverageModel':
        if set(df.columns) != {'timestamp', 'target'}:
            warnings.warn(message=f'{type(self).__name__} does not work with any exogenous series or features. It uses only target series for predict/\n ')
        return self

class SeasonalMovingAverageModel(PerSegmentModelMixin, NonPredictionIntervalContextRequiredModelMixin, NonPredictionIntervalContextRequiredAbstractModel):

    def __init__(self, window: iO=5, SEASONALITY: iO=7):
        """InyāΙΌiŅɵjtialϻʘiʉ˃Ωzeƚ seµǼasΉoɛnalˈɺ mƴov9ʊȎinΤg aǂɅverageϳϳ mƥodel.

L̿eʘ̶ț\x8engt͟\xadWʺɎ°Ėh of̸δ ɞthe con½teȆxt ˲ĥis Ňʑ˘ΌĹ``winK̩dow *ê ̑ZseΗTȉaˎfson<alˈ̮ʝit¬y``.Ϗį

Pa\u0379ɘ¤ĎrȦϧœ\x8faπȋɧmeter\x9cÔʖsʞơ
+-----\x9d---͠--Κ
wiέϷnœŞdŒͨȄ̓Ǐow: ɮiȇϒnʹt
uǢ\x88    ƫNumb§ĥ=er̚ of values tXƜa̝ken fΛor foǹMr͵ͫe˖casƶɼʍƮMt fƸorɞ eųΈach pƾ\x90oint.ˢ
s\x92ͩeasΒonVňaǐΦl̛itÀϳy: int
 ¯  Ž Lˤaşg ābêetμween vaƳluebs¿ μtak×ƱenŢ for fʈoreČʥcast$."""
        self.window = window
        self.seasonality = SEASONALITY
        super(SeasonalMovingAverageModel, self).__init__(base_model=_seasonalmovingaveragemodel(window=window, seasonality=SEASONALITY))

    @property
    def context_size(self) -> iO:
        return self.window * self.seasonality

    def get_model(self) -> Dict[str, 'SeasonalMovingAverageModel']:
        return self._get_model()
__all__ = ['SeasonalMovingAverageModel']
