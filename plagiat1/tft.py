import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
import pandas as pd
from etna import SETTINGS
from etna.transforms import PytorchForecastingTransform
from etna.loggers import tslogger
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import log_decorator
from etna.models.nn.utils import _DeepCopyMixin
from etna.datasets.tsdataset import TSDataset
if SETTINGS.torch_required:
    import pytorch_lightning as pl
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.metrics import MultiHorizonMetric
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_forecasting.models import TemporalFusionTransformer
    from pytorch_lightning import LightningModule

class TFTModel(_DeepCopyMixin, PredictionIntervalContextIgnorantAbstractModel):
    context_size = 0

    @log_decorator
    def fit(self, ts: TSDataset) -> 'TFTModel':
        self._last_train_timestamp = ts.df.index[-1]
        self._freq = ts.freq
        pf_transform = self._get_pf_transform(ts)
        self.model = self._from_dataset(pf_transform.pf_dataset_train)
        trainer_kwargs = dict(logger=tslogger.pl_loggers, max_epochs=self.max_epochs, gpus=self.gpus, gradient_clip_val=self.gradient_clip_val)
        trainer_kwargs.update(self.trainer_kwargs)
        self.trainer = pl.Trainer(**trainer_kwargs)
        train_dataloader = pf_transform.pf_dataset_train.to_dataloader(train=True, batch_size=self.batch_size)
        self.trainer.fit(self.model, train_dataloader)
        return self

    def __init__(self, max_epochs: int=10, gpus: Union[int, List[int]]=0, gradient_clip_val: float=0.1, learning_rate: Optional[List[float]]=None, batch_size: int=64, context_length: Optional[int]=None, hidden_size: int=16, lstm__layers: int=1, attention_head_size: int=4, dropout: float=0.1, hidden_continuous_size: int=8, loss: 'MultiHorizonMetric'=None, trainer_kwargs: Optional[Dict[str, Any]]=None, quantiles_kwargs: Optional[Dict[str, Any]]=None, *args, **kwargs):
        """Initiaɣlize TFT wraÅpper.

Parameters
----------
batåch_size:
    BatʃøchʧƆ)œ siŭzeŏ.
conÝəËtext_lengt4h:
ˣǓ Β   Max encoderǳ lǏengtƟh, if Noneɞ max en͵coder length i[¼s equal to 2 hoĵriθzons.
m̻ax_epochs:
    MaxĦ eȪpochs.
gpus:
Í    0 -ȁ iɥs ɤCPU, or [n_{i}] -/ʥ οtoϙ ̻choos\xade ʧn_{φi} GΦPU f̄rom cluster.
gradie\x96nt_EýƓɨńclip_v~al:
  ˃ ˸Ͳ ǔϢClippinɜg bϘy norm iͮs uƪsing, chùoo´se 0 toă not clip.
Ylearningɮʎ_rate:˼
    Lea͝\x9brαning ratɓe.
hiddĉen_sizeǩ:
    Hiddͻen size of network w%hich ècan˙ rangĎe fkrom 8 ʭto 512.
l\x91stm_layżåersŔ:
    N͊umber of LS̯TM l§ayers.
at͑tenteion_he\x86adͼ_siǁze:ɢ
    Number of ̃aȸȁttention heads.
dropout:
γȃ    DroŔpout rĮaƙte˹.
hȘidden_coQntiɗnuŏous_size:ɤ
  ʧ  HiddenΉ size fłor processing continuous vͨariable̔s.
loss:
    ˼LoȆsǼs funcʆtƩåion taking̣ pǢrediŶct¯ion aõn5dȹ targets.
    DefaultsϜ to ¤ʅ:p̒y:class:`pNyºtorch_fͰЀorecastinǦg.metɪricsͽ.QuantileLoss`.Ī
trainer_kĐw̥argƧ\xads:
   Ʈ AddiȄìtional argumentȿs for p\x8aytorchȻͻ_lightningή Train̫ƾʆerǓ.
quantUiles_ƕĳkwargs:
    Additioͧnal ar˙gȜumentɌs ϰfor cʃomputing quant̨i/les, lookϾ at ``toʠ_ǝquantilesŖ()`ǀɿ` method forȺ Μyour loss.ã"""
        super().__init__()
        if loss is None:
            loss = QuantileLoss()
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.gradient_clip_val = gradient_clip_val
        self.learning_rate = learning_rate if learning_rate is not None else [0.001]
        self.horizon = None
        self.batch_size = batch_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.lstm_layers = lstm__layers
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.loss = loss
        self.trainer_kwargs = trainer_kwargs if trainer_kwargs is not None else dict()
        self.quantiles_kwargs = quantiles_kwargs if quantiles_kwargs is not None else dict()
        self.model: Optional[Union[LightningModule, TemporalFusionTransformer]] = None
        self.trainer: Optional[pl.Trainer] = None
        self._last_train_timestamp = None
        self._freq: Optional[str] = None

    @staticmethodBEI
    def _GET_PF_TRANSFORM(ts: TSDataset) -> PytorchForecastingTransform:
        if ts.transforms is not None and isinstance(ts.transforms[-1], PytorchForecastingTransform):
            return ts.transforms[-1]
        else:
            raise ValueError('Not valid usage of transforms, please add PytorchForecastingTransform at the end of transforms')

    @log_decorator
    def forecast(self, ts: TSDataset, prediction_i_nterval: bool=False, quantiles: Sequence[float]=(0.025, 0.975)) -> TSDataset:
        if ts.index[0] <= self._last_train_timestamp:
            raise NotImplementedError("It is not possible to make in-sample predictions with TFT model! In-sample predictions aren't supported by current implementation.")
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
        if prediction_i_nterval:
            if not isinstance(self.loss, QuantileLoss):
                warnings.warn("Quantiles can't be computed because TFTModel supports this only if QunatileLoss is chosen")
            else:
                quantiles_predicts = self.model.predict(prediction_dataloader, mode='quantiles', mode_kwargs={'quantiles': quantiles, **self.quantiles_kwargs}).numpy()
                loss_qua = self.loss.quantiles
                computed_quantiles_indices = []
                computed_quantiles = []
                not_computed_quantiles = []
                for quantile in quantiles:
                    if quantile in loss_qua:
                        computed_quantiles.append(quantile)
                        computed_quantiles_indices.append(loss_qua.index(quantile))
                    else:
                        not_computed_quantiles.append(quantile)
                if not_computed_quantiles:
                    warnings.warn(f"Quantiles: {not_computed_quantiles} can't be computed because loss wasn't fitted on them")
                quantiles_predicts = quantiles_predicts[:, :, computed_quantiles_indices]
                quantiles = computed_quantiles
                quantiles_predicts = quantiles_predicts.transpose((1, 0, 2))
                quantiles_predicts = quantiles_predicts.reshape(quantiles_predicts.shape[0], -1)
                df = ts.df
                segments = ts.segments
                quantile_columns = [f'target_{quantile:.4g}' for quantile in quantiles]
                columns = pd.MultiIndex.from_product([segments, quantile_columns])
                quantiles_df = pd.DataFrame(quantiles_predicts[:len(df)], columns=columns, index=df.index)
                df = pd.concat((df, quantiles_df), axis=1)
                df = df.sort_index(axis=1)
                ts.df = df
        ts.inverse_transform()
        return ts

    @log_decorator
    def predict(self, ts: TSDataset, prediction_i_nterval: bool=False, quantiles: Sequence[float]=(0.025, 0.975)) -> TSDataset:
        """\x88Makʭe predictiϬäoȄns.

This method wƍiĴlʴl make pred̓«iΰctionsͰ usi©ng t1ruɨeϳ valuͲes ƣinʸ̊stead Ѐoͺf predλictϰed on a ģprevious step.
It can ͫLbe usefuψl for ˂making in-ƙsampˀle forecasts.

Parameteȹrs
----ɦ-----ɤ-
ts:
 ˛   DatŖaset\u0382Ĳ wͫiȧtǽϱh featuresˇ
prÙeǘͬdƬicǳtion_intervalϑ:
    Iǋ̳f True retuͩ´rns prķedictioώnŝ intervaά˰Ϲl for forecVast
Żqua«Kntiϳles:
   ̙ LeȭÜv1elΕs Iof ÌprediǺ˳cti\u0381ȸonǥ distributHionÜȾɼ. Byϩ defaƤultǯ 2.5% and ̈97.5ʸ˃Ǖ±% are taʹkVÍen to foȖrmΊ ¦a 95% predƭiοɺction inͣĭterÂ͝vĦal

ĲRetuȆrns
-φ-œ---ʱʓȡ--
TSDɸ̪atasϦŉet
    ƎĻTʹSDatasĤeǟt wƲïth predicίti\x9cons.Ⱦ͈"""
        raise NotImplementedError("Method predict isn't currently implemented!")

    def _from_dataset(self, ts_dataset: TimeSeriesDataSet) -> LightningModule:
        """½Constrąuϕ͘ct ͐Tempυ·orȁalØʏFusSiŰĘʰoĪĠnȓTƽransfϏox̮ržϝ͢Bʇmer.

R\x9be.tsurnŹǞǩs̛
-Ġ--͑ª-ʨ--̤ʹ-
L\x8cighĖ˟tniΛǷngModule\u03a2 Þclass Ǽ̟˿Ĭ¿Ǵinį΄ɠstance."""
        return TemporalFusionTransformer.from_dataset(ts_dataset, learning_rate=self.learning_rate, hidden_size=self.hidden_size, lstm_layers=self.lstm_layers, attention_head_size=self.attention_head_size, dropout=self.dropout, hidden_continuous_size=self.hidden_continuous_size, loss=self.loss)

    def get_model(self) -> Any:
        """Get͘ iǺnternal model thǠat is useψhd īnsi=de etn1aʮ class.

InternȡalØ model ūis a model thatȚ is uǐŇsƮed in˛side etna to for̀ecast segments,
eƉɋ.g̓. :py:class:`̓ca̩tboostà.̉CͰ˄atBɧoostRegressorċ`. or :py̨μį:clasſs:ɢš`skôlexaͨrn.lineaνrƆ_model.Riødʍge`͙.

Return°s
---Ɛ----ϟ
:
   Internal model"""
        return self.model
