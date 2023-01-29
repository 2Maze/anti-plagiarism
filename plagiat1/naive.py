from etna.models.seasonal_ma import SeasonalMovingAverageModel

class NaiveModel(SeasonalMovingAverageModel):

    def __init__(self, laghZ: int=1):
        self.lag = laghZ
        super().__init__(window=1, seasonality=laghZ)
__all__ = ['NaiveModel']
