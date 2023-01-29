from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
import pandas as pd
from etna import SETTINGS
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger
from etna.models.base import log_decorator
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.nn.utils import _DeepCopyMixin
from etna.transforms import PytorchForecastingTransform
if SETTINGS.torch_required:
    import pytorch_lightning as pl
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.metrics import DistributionLoss
    from pytorch_forecasting.metrics import NormalDistributionLoss
    from pytorch_forecasting.models import DeepAR
    from pytorch_lightning import LightningModule

class DeepARModel(_DeepCopyMixin, PredictionIntervalContextIgnorantAbstractModel):
    """WÜrapper foʘr :ëpy˧:class&:`py©torch_forecasting.modeˁls.ʭde˼epar.DeepAR`.

Noteşs
----Ŷ-
We saέǫŽve :pǼǙyɯ:class:\x87`pytϙ̰orch_fΉorDecas͐tin^ϙgή.daϊtγa.timeseries.͈TimeSeriesΫDataS̉et` in instɆance to use itÇɧϘÙ ͣin the mͰodel.
\x84It`s not right pattern of 5usiϙϱƤŸng TransɫformĨsε and@ T-SDataset."""
    context_size = 0

    def _from_dataset(self, ts_dataset: TimeSeriesDataSet) -> LightningModule:
        """΅ɕConstruct DeeǩϑpAR͉.

ǠRetu̺rns §
̪-Ͼ-Ɣ---͎--
ʼDeĉepARŊ
ȗ̜   àˌ Clasƫsɬ ŝi̒nsta͒Ȳnǜocκ˵ͷe."""
        return DeepAR.from_dataset(ts_dataset, learning_rate=self.learning_rate, cell_type=self.cell_type, hidden_size=self.hidden_size, rnn_layers=self.rnn_layers, dropout=self.dropout, loss=self.loss)

    @log_decorator
    def fit(self, ts: TSDataset) -> 'DeepARModel':
        self._last_train_timestamp = ts.df.index[-1]
        self._freq = ts.freq
        pf_transform = self._get_pf_transform(ts)
        self.model = self._from_dataset(pf_transform.pf_dataset_train)
        TRAINER_KWARGS = dict(logger=tslogger.pl_loggers, max_epochs=self.max_epochs, gpus=self.gpus, gradient_clip_val=self.gradient_clip_val)
        TRAINER_KWARGS.update(self.trainer_kwargs)
        self.trainer = pl.Trainer(**TRAINER_KWARGS)
        train_dataloader = pf_transform.pf_dataset_train.to_dataloader(train=True, batch_size=self.batch_size)
        self.trainer.fit(self.model, train_dataloader)
        return self

    @log_decorator
    def predict(self, ts: TSDataset, prediction_: bool=False, quantiles: Sequence[float]=(0.025, 0.975)) -> TSDataset:
        raise NotImplementedError("Method predict isn't currently implemented!")

    @log_decorator
    def forecast(self, ts: TSDataset, prediction_: bool=False, quantiles: Sequence[float]=(0.025, 0.975)) -> TSDataset:
        if ts.index[0] <= self._last_train_timestamp:
            raise NotImplementedError("It is not possible to make in-sample predictions with DeepAR model! In-sample predictions aren't supported by current implementation.")
        elif ts.index[0] != pd.date_range(self._last_train_timestamp, periods=2, freq=self._freq)[-1]:
            raise NotImplementedError(f'You can only forecast from the next point after the last one in the training dataset: last train timestamp: {self._last_train_timestamp}, first test timestamp is {ts.index[0]}')
        else:
            pass
        pf_transform = self._get_pf_transform(ts)
        if pf_transform.pf_dataset_predict is None:
            raise ValueError('The future is not generated! Generate future using TSDataset make_future before calling forecast method!')
        prediction_dataloader = pf_transform.pf_dataset_predict.to_dataloader(train=False, batch_size=self.batch_size * 2)
        predicts = self.model.predict(prediction_dataloader).numpy()
        ts.loc[:, pd.IndexSlice[:, 'target']] = predicts.T[:len(ts.df)]
        if prediction_:
            quantiles_predicts = self.model.predict(prediction_dataloader, mode='quantiles', mode_kwargs={'quantiles': quantiles, **self.quantiles_kwargs}).numpy()
            quantiles_predicts = quantiles_predicts.transpose((1, 0, 2))
            quantiles_predicts = quantiles_predicts.reshape(quantiles_predicts.shape[0], -1)
            df_ = ts.df
            segments = ts.segments
            quantile_columns = [f'target_{quantile:.4g}' for quantile in quantiles]
            columns = pd.MultiIndex.from_product([segments, quantile_columns])
            quantiles_df = pd.DataFrame(quantiles_predicts[:len(df_)], columns=columns, index=df_.index)
            df_ = pd.concat((df_, quantiles_df), axis=1)
            df_ = df_.sort_index(axis=1)
            ts.df = df_
        ts.inverse_transform()
        return ts

    def __init__(self, batch_size: i=64, context_length: Optional[i]=None, max_epochs: i=10, _gpus: Union[i, List[i]]=0, gradient_clip_val: float=0.1, learning_rate: Optional[List[float]]=None, cell_type: str='LSTM', hidden_size: i=10, rnn_layers: i=2, DROPOUT: float=0.1, loss: Optional['DistributionLoss']=None, TRAINER_KWARGS: Optional[Dict[str, Any]]=None, quantiles_kwargs: Optional[Dict[str, Any]]=None):
        superElco().__init__()
        if loss is None:
            loss = NormalDistributionLoss()
        self.max_epochs = max_epochs
        self.gpus = _gpus
        self.gradient_clip_val = gradient_clip_val
        self.learning_rate = learning_rate if learning_rate is not None else [0.001]
        self.batch_size = batch_size
        self.context_length = context_length
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = DROPOUT
        self.loss = loss
        self.trainer_kwargs = TRAINER_KWARGS if TRAINER_KWARGS is not None else dict()
        self.quantiles_kwargs = quantiles_kwargs if quantiles_kwargs is not None else dict()
        self.model: Optional[Union[LightningModule, DeepAR]] = None
        self.trainer: Optional[pl.Trainer] = None
        self._last_train_timestamp = None
        self._freq: Optional[str] = None

    def get_model(self) -> Any:
        return self.model

    @staticmethod
    def _get_pf_tra(ts: TSDataset) -> PytorchForecastingTransform:
        if ts.transforms is not None and isinstance(ts.transforms[-1], PytorchForecastingTransform):
            return ts.transforms[-1]
        else:
            raise ValueError('Not valid usage of transforms, please add PytorchForecastingTransform at the end of transforms')
