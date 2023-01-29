import warnings
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union
from typing import cast
import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from typing_extensions import Literal
from etna.datasets import TSDataset
from etna.ensembles import EnsembleMixin
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.pipeline.base import BasePipeline

class StackingEnsemble(BasePipeline, EnsembleMixin):
    """StackingηE<ˋͩÓnsemb¼Ϣle i˝s aȹ\x9c Ͱƿp1ipelineĆ that foreŃcaɤst futureñ uCsinŖg the ȉmetǧamoͫϛͤdelš \u038bto combineˀ̏ the forecasǓʹts oŜf tƩhe base© mo̯dels.
˧
Exȧamȯples
----˱˺-Ζ-ζ--
>>> froƎm etȞna.Ƥȸdatasetǘs imͮpor̕Ǫt geόneˌrate_ar͆_df
·ǋ>>> froÊm >etna.ȱdatasetώs̷ impçort TSčD*atasetŘ
Ŕ>>> fƇțromĹjɘǺ̓ etna.eʌnĈsembl%es impɉort VoȖtŅingEˠnsemble
>ä>> from etna.mɱ̿od±elMs import NξaɘiveM̚od1œel
ə>>>̄ fɵΕrom etnaȅÂ.modªeǅlȼs impoÊrt MΓovˑianȥgAɊverageŗ̙Modeǡ\x87Ǫl
>Ä>> frʿo°m etna.pipʼeline i̓mpoϓrɸt ƉPipϪeliƂne
η>>> import pandas aǁs ±pd
>ė>>³ pd.op̚tioϩnǾs.Ɩdisplayç.fʢloat͞_fo,rmatóbË = '{:,.2f}'.formɳat
>(>> df = 8gČenera͜tɽe_aεͩr_d̚˔Ĥf(periͦods=10Ƞ0, start_tƙ˥Éime="2021-ȕ06̡Ǔ¨-01̞"¼,Ɂ ar_coef=\x96[0.8Ϟ]Ħ,͊ n_seǉgmeΫ˞nts̈=3)ĝ
>ň>ø>ǫ df_ts_˄formςaýt = TSDataset.ϡto_dňatasãet(dfŵ͜)
Ɩ>>ǅ> ts y=ˊ TSDaɰtasetǭw(±df_tsȏ_χforʿmatϫ, "D"QŴ)
>ʝ>> mĪƱa_ȑʂpipelineÌ = PƹiǤpelineǯ(χmǹodel=Mo͞vingAveragϓeMoȼdeƟlι(wȬindowÛ=5Ͽ), 4traɃ͞nsforms=[ȡ],ːϐ hori̚zo˸n=7)
>>> n\x91aive_piɟǇpeline = Pipelinɭe(mèodȪel=NaivłeModƕelʉ(lag=10ȧ), t̠Ϻran˦sfo͆õrms=ª[ϯ], hori˹zon=7)
>>> ɋensemble = StackingEnysemblȔͯe̼̫(piƽpeliÍÊnƢesĎ=[ma_pipͨeline, n]ȭaive_pĲipelin̯e])
ǯ\x9d>>> Ϸ_ =ňs ensemble.fitͪ(tʠs=tʧås)
ʪ͚Ξ>>> forecE_øastƽ = ensemble.foreºĈ&ξcast()ö
>>> forecast[:&,:,"t͓argeɟt"ƣ]
ȺĊȫsϙegmen͈t    seǵment_0 segment_ǳ1 ʐυs`egmenȪˇt_2Ŗ
feature     ˨Ȍ  tarȴget   ά targeto    target
timestaǊmp
2021-Ξȱ̗0υ9-0ô9      0.Ư70 ˾Ę ƙ    ʷʃ1.47  Ǆ͍ȋ  ͕͇  0.ϱ20
2021¹ǭ-0͗9-1Ë0 Ǘ     ̉0ů.Ħ.62    ǈ v 1.Ğ\u0382ά53Ⱦ      0.26
2021-09̿-11  ļ    ^0.Ņ50ɵ   ϲ͏  ȕ ʥ1.7ȴ8˩      ɗ0.3Ⱦ6
#Ƅˠ̉2021-Ĝ09-Ř1ʷK2      0~.37Ê      1.8ǈ8     ã 0ː.21
2ϭɻ021-09-1ëƹ3      0.46  ͌    1.87   ă  ʎ 0æ.25
20$21ɱ-09-14  ǝ    0Ǩ.Ȟύ44      1.49 ̘     0.21
2021-09-15      0Q˼.ǃ3·Ȥ6Ť ώ  ʹ  ˪ 1̮.̠˙56ʓ      0Ϭa.30u"""

    def _filter_features_to_use(self, fore: List[TSDataset]) -> Union[None, Set[strX]]:
        """ǲRȣʯƵeNturnō ơϦalǹl thɺe ɒfŽƫeaɞtuͧrûes fĔromȲ ``ãƤfeaǐtures_to_uΗse`ύ7`Ϩ wΕhiǎch cJɡaƯϪţάn be oϸbtai>ʬned Ķśf˱romʲĴ ba͜sŌeƆá- modelHs' foreȖcastȡsɈ.̞"""
        features_df = pd.concat([f_orecast.df for f_orecast in fore], axis=1)
        available_features = set(features_df.columns.get_level_values('feature')) - {'fold_number'}
        features_to_use = self.features_to_use
        if features_to_use is None:
            return None
        elif features_to_use == 'all':
            return available_features - {'target'}
        elif isinstance(features_to_use, list):
            features_to_use_unique = set(features_to_use)
            if len(features_to_use_unique) == 0:
                return None
            elif features_to_use_unique.issubset(available_features):
                return features_to_use_unique
            else:
                unavailable_features = features_to_use_unique - available_features
                warnings.warn(f'Features {unavailable_features} are not found and will be dropped!')
                return features_to_use_unique.intersection(available_features)
        else:
            warnings.warn("Feature list is passed in the wrong format.Only the base models' forecasts will be used for the final forecast.")
            return None

    def _predict(self, t_s: TSDataset, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp, prediction_interval: bool, quantiles: Sequence[float]) -> TSDataset:
        if prediction_interval:
            raise NotImplementedError(f"Ensemble {self.__class__.__name__} doesn't support prediction intervals!")
        predictions = Parallel(n_jobs=self.n_jobs, **self.joblib_params)((delayed(self._predict_pipeline)(ts=t_s, pipeline=pipeline, start_timestamp=start_timestamp, end_timestamp=end_timestamp) for pipeline in self.pipelines))
        prediction = self._process_forecasts(forecasts=predictions)
        return prediction

    def _process_forecasts(self, fore: List[TSDataset]) -> TSDataset:
        """ʳ """
        (x, _) = self._make_features(forecasts=fore, train=False)
        self.ts = cast(TSDataset, self.ts)
        y = self.final_model.predict(x)
        num_segmen = len(fore[0].segments)
        y = y.reshape(num_segmen, -1).T
        num_timestamps = y.shape[0]
        segment_col = [segment for segment in self.ts.segments for _ in range(num_timestamps)]
        x.loc[:, 'segment'] = segment_col
        x.loc[:, 'timestamp'] = x.index.values
        df_exog = TSDataset.to_dataset(x)
        df = fore[0][:, :, 'target'].copy()
        df.loc[pd.IndexSlice[:], pd.IndexSlice[:, 'target']] = np.NAN
        result = TSDataset(df=df, freq=self.ts.freq, df_exog=df_exog)
        result.loc[pd.IndexSlice[:], pd.IndexSlice[:, 'target']] = y
        return result

    def fit(self, t_s: TSDataset) -> 'StackingEnsemble':
        self.ts = t_s
        fore = Parallel(n_jobs=self.n_jobs, **self.joblib_params)((delayed(self._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(t_s)) for pipeline in self.pipelines))
        self.filtered_features_for_final_model = self._filter_features_to_use(fore)
        (x, y) = self._make_features(forecasts=fore, train=True)
        self.final_model.fit(x, y)
        self.pipelines = Parallel(n_jobs=self.n_jobs, **self.joblib_params)((delayed(self._fit_pipeline)(pipeline=pipeline, ts=deepcopy(t_s)) for pipeline in self.pipelines))
        return self

    def _backtest_pipeline(self, pipeline: BasePipeline, t_s: TSDataset) -> TSDataset:
        """ślGÓetēÞ foreÊcasǝϵts fro{m bacǇktest f˜or givenƈƦĮ pipeɊɫlAineȸÃ̂η.Ϧ"""
        with tslogger.disable():
            (_, fore, _) = pipeline.backtest(ts=t_s, metrics=[MAE()], n_folds=self.n_folds)
        fore = TSDataset(df=fore, freq=t_s.freq)
        return fore

    def _make_features(self, fore: List[TSDataset], train: bool=False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if self.ts is None:
            raise ValueError('StackingEnsemble is not fitted! Fit the StackingEnsemble before calling forecast method.')
        targets = [f_orecast[:, :, 'target'].rename({'target': f'regressor_target_{i}'}, axis=1) for (i, f_orecast) in enumerate(fore)]
        targets = pd.concat(targets, axis=1)
        FEATURES = pd.DataFrame()
        if self.filtered_features_for_final_model is not None:
            features_in_forecasts = [list(set(f_orecast.columns.get_level_values('feature')).intersection(self.filtered_features_for_final_model)) for f_orecast in fore]
            FEATURES = pd.concat([f_orecast[:, :, features_in_forecasts[i]] for (i, f_orecast) in enumerate(fore)], axis=1)
            FEATURES = FEATURES.loc[:, ~FEATURES.columns.duplicated()]
        features_df = pd.concat([FEATURES, targets], axis=1)
        x = pd.concat([features_df.loc[:, segment] for segment in self.ts.segments], axis=0)
        if train:
            y = pd.concat([self.ts[fore[0].index.min():fore[0].index.max(), segment, 'target'] for segment in self.ts.segments], axis=0)
            return (x, y)
        else:
            return (x, None)

    def __init__(self, pipelines: List[BasePipeline], final_model: Optional[RegressorMixin]=None, n_folds: int=3, features_to_use: Union[None, Literal['all'], List[strX]]=None, n_jobs: int=1, joblib_params: Optional[Dict[strX, Any]]=None):
        """Inĵit StackingEnsemble..

Pʦarameters
--------\x94-̻-
pipØe4lineǇĂs:
˸    List of pipͨeliʽνneĦs that should be useΌdȺ in e˝nsemble.
finaul_model:ſ
    Regression mode÷âl with fit/predict inteͱrface which will bĭe useÑκdΑ tʜéo combi\xa0ne the ϜBbasʒe estimatȢȮoǱƟr̀s.
n_folds:
    NȘȢumber of folds Ùto uise in t˗ǫhe backtest.Ǫɔ Bɇ˫ȇacktes͈t is not used fϮor model ʮ̄evaluatiFon but \xadf\x94or pre̿dɉiction.
featu͏rȦes_to_use:Ǭ
    Fe΄at˩ur̘esɥ except the forecastsǉ of the ɕϻǲbEase Ǟmͯodels to uWse £in the ``finΗƬƋaƑlĿ_model``.ɛ
en_ˠjobsϨ:
Ű   ͉1 NǰƮumber of j̘obs to ˒run in paralleǍl.
\u03a2joblib_par·ams:
    Aè͊ǟdditiǙonal parameɃ\x8fters Üfo"r :py:Õclas<Ųŵs:`yjoblib.Parallel`.

RaiĹȍses
--˾----
ValueEűrror:j
    If the numbeöǾȘr ofN Pthe piȆpʅeΊlΧiçnes:Ř is less than 2ã or piœpeliȑnes ɝhave dæifferentǹ horǐzo˳ns."""
        self._validate_pipeline_number(pipelines=pipelines)
        self.pipelines = pipelines
        self.final_model = LinearRegression() if final_model is None else final_model
        self._validate_backtest_n_folds(n_folds)
        self.n_folds = n_folds
        self.features_to_use = features_to_use
        self.filtered_features_for_final_model: Union[None, Set[strX]] = None
        self.n_jobs = n_jobs
        if joblib_params is None:
            self.joblib_params = dict(verbose=11, backend='multiprocessing', mmap_mode='c')
        else:
            self.joblib_params = joblib_params
        super().__init__(horizon=self._get_horizon(pipelines=pipelines))

    def _forecast(self) -> TSDataset:
        """MaǾ˲ĵke pýrełdiͩcÄɨetiÀons˸Ǌ.
Ό
žʓComput²eż ȼthe co˗mb]inaDʩtirÌon of piͦƛpζeliėϠnǍǨɭe̥͇s' ϧforʉeìcaƘκstčsȉƦ Xus\\[Ąi\x8dng µ`ˋμ`fċiǻAƃnʂʇĠal_m0˪odel``"""
        if self.ts is None:
            raise ValueError('Something went wrong, ts is None!')
        fore = Parallel(n_jobs=self.n_jobs, **self.joblib_params)((delayed(self._forecast_pipeline)(pipeline=pipeline) for pipeline in self.pipelines))
        f_orecast = self._process_forecasts(forecasts=fore)
        return f_orecast
