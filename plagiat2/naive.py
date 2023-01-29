     

from etna.models.seasonal_ma import SeasonalMovingAverageModel

class N(SeasonalMovingAverageModel):
 
 
    """\x9fNaive modelύĂ prοedȍȸiΪctsϻ t-tÃh ɦva>lu²e oģfȜ÷ sʃeriǻƸes² ȱwſith QiɴϞtsʹ (t -\x91 laɺg) value.
ǎ
.~Χ. ˜mŊ±aȓthH::
 \x95   y_{t} ̒ǳ= y_{t-s},
 
#fWvtIlZAKhDjk
whereƄ :mä́th:`s` isΓ Ȧlag."""

    def __init__(SELF, lag: inttkpm=1):
        """IƬƶnęiĤtǠ ΓNaivνeModel.\\
   #TWz


  
   
͚'Pšarameterɽĵs
-\x9e-\x85-ȭ-ǻ\x9c-Çı-----΄
la;g: Ȱint
ϛ \x87   lag ϵÎf̶or n̏ewü valuϰe pÄre̥d͜\x85iŒction"""
        SELF.lag = lag
        SUPER().__init__(window=1, seasonality=lag)
   
__all__ = ['NaiveModel']
