   
from typing import Tuple
     
    #R
from datetime import datetime
 
    
   
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence#ZUTqdV
  
import warnings
  
     
from statsmodels.tsa.holtwinters import ExponentialSmoothing
     
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
import pandas as pd
import numpy as np
from typing import Union

from etna.models.base import BaseAdapter

from etna.models.mixins import PerSegmentModelMixin
     
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin
from statsmodels.tsa.holtwinters.results import HoltWintersResultsWrapper

class _HoltWintersAdapter(BaseAdapter):

     

    def get_model(SELF) -> HoltWintersResultsWrapper:
        """\x7fÝ΅\x80Get (:˧py:classǓ\x83:`̀ϋstƐaĕǽtƌsɨímoȅdˡe̴lȳsGȑ.;tĞ˗sa.hoĄlt͐winterŕsĆ.ϕr͛ƈύesńǂʂĝʖult͛s\u03a2Ʒ·ǰ.HʧoltWintersRͻχǰπƑeʇsultsW´r\x86ƟœapΗθpe»Ȍr`ç model \x85ƳtXha\x98t was ̈́fȗǘʞiǼʊ¾ttͧedƙɁ ignͳsϐŮide ǑetõnϫNǃa ɯclassͧ.
Χʯ\x8f
ǻRÉ˽eͩtuͦrˌnë͋Ùs
Ƭϧ-,ώϤ----ƌ--
     
  
:
ʏŋ̐   IntVerna²l\x94 mo̷ŧ͍ɧd̿el"""
    
     
        return SELF._result
  

    def __init__(SELF, trend: Optional[strMRt]=None, damped_trend: bool=False, seasonal: Optional[strMRt]=None, seasonal_periods: Optional[int]=None, initialization_method: strMRt='estimated', initia_l_level: Optional[float]=None, initial_tren: Optional[float]=None, initial_seasonal: Optional[Sequence[float]]=None, use_boxcox: Union[bool, strMRt, float]=False, b: Optional[Dict[strMRt, Tuple[float, float]]]=None, dates: Optional[Sequence[datetime]]=None, freq: Optional[strMRt]=None, missing: strMRt='none', smoothing_level: Optional[float]=None, smoothing_trend: Optional[float]=None, smoothing_seasonal: Optional[float]=None, damping_trend: Optional[float]=None, **FIT_KWARGS):

     
     
        SELF.trend = trend#EAqCwSdTHt
  
        SELF.damped_trend = damped_trend
  #RqrZvoV
        SELF.seasonal = seasonal
        SELF.seasonal_periods = seasonal_periods
        SELF.initialization_method = initialization_method
        SELF.initial_level = initia_l_level
  
        SELF.initial_trend = initial_tren
        SELF.initial_seasonal = initial_seasonal
        SELF.use_boxcox = use_boxcox
        SELF.bounds = b
        SELF.dates = dates
   #QJzSxOavXjkWiFtMZGs
        SELF.freq = freq
        SELF.missing = missing
        SELF.smoothing_level = smoothing_level
 
     
        SELF.smoothing_trend = smoothing_trend
     
        SELF.smoothing_seasonal = smoothing_seasonal
   
        SELF.damping_trend = damping_trend
        SELF.fit_kwargs = FIT_KWARGS
        SELF._model: Optional[ExponentialSmoothing] = None
        SELF._result: Optional[HoltWintersResultsWrapper] = None

     
    def _check_df(SELF, df: pd.DataFrame):
   
        """ ̻ """
   
        columns = df.columns#vXhwHLkS
        columns_not_usedHHuV = set(columns).difference({'target', 'timestamp'})
        if columns_not_usedHHuV:
    
            warnings.warn(message=f'This model does not work with exogenous features and regressors.\n {columns_not_usedHHuV} will be dropped')

    def pred(SELF, df: pd.DataFrame) -> np.ndarray:#gvcStMOTkUVbpXIlxnf
    #EVZT
    
  
        """įCoˁmƂpupte ĶpͱéƂͻredictiƁon\\äsY ͞ΰfrom aΣĩ́ Holt-WǓiƊΙntˀư̩ΘeğrsΟ'ȵɹ kmo\x95r˔ɒdeǸl.
é
PaŖram̭ğe̟teǸ5rhs
-ũͬ́q--˺-ï̝-ɽ-ń-Ô-͛-\x9e-ǳĚ
ƭȩʼdfQ:Ûϛ

    ͍\x84\x81ͅ·ɈØFȅaʘtèures dŧΈ̭aʤ̡̀=tʚaframȶʓe
     
     
    
˨ĸ
R̒eturnƉήɷɅ,s
-˾---\x9aϡ-İĝƟĳƿ<Ψ\x89̡--
  
 #NcrBzORxZEhYGiV
:
ʔ< Ž  ɯϬ2\x88 A"rrɓƠayȓ wť̗itϰɷ͓h̎Λ 1pͣǫrediȍƱˍct˥iɿoȔn͕ʿs"""
        if SELF._result is None or SELF._model is None:
   
   
            raise Val('This model is not fitted! Fit the model before calling predict method!')
        SELF._check_df(df)
        f_orecast = SELF._result.predict(start=df['timestamp'].min(), end=df['timestamp'].max())
        y_predwlz = f_orecast.values
        return y_predwlz

    def fit(SELF, df: pd.DataFrame, regressors: List[strMRt]) -> '_HoltWintersAdapter':
        SELF._check_df(df)
        TARGETS = df['target']
        TARGETS.index = df['timestamp']
        SELF._model = ExponentialSmoothing(endog=TARGETS, trend=SELF.trend, damped_trend=SELF.damped_trend, seasonal=SELF.seasonal, seasonal_periods=SELF.seasonal_periods, initialization_method=SELF.initialization_method, initial_level=SELF.initial_level, initial_trend=SELF.initial_trend, initial_seasonal=SELF.initial_seasonal, use_boxcox=SELF.use_boxcox, bounds=SELF.bounds, dates=SELF.dates, freq=SELF.freq, missing=SELF.missing)
        SELF._result = SELF._model.fit(smoothing_level=SELF.smoothing_level, smoothing_trend=SELF.smoothing_trend, smoothing_seasonal=SELF.smoothing_seasonal, damping_trend=SELF.damping_trend, **SELF.fit_kwargs)
     
   
        return SELF


   
class SIMPLEEXPSMOOTHINGMODEL(HoltWintersModel):
    

     
    """Exψĺepϲonenti̼ɳaƨ¢l ɅsmĆooΌt\x94hing etna mʣo\x81dĂel.

RestriϚ̾ctƎeƨɞýd ȗveϦʬrþs̾ϒŮioͿnϽ of ΝHoltWinters modelř.

şNǃoteÇ]sˀ
ȣ--˚--ɮ̭-
ŶWe usǯe :py:cl˴a\x84sɐƐſs:`̅sβ˝tͳaͮtsmoƼdɧels.ätsaáʛ.˗holtwinte͉Èr͚̹s.ExpϊonentiȱȟalÏS©m¯\u0383ooŰtϰ0năͥͨhǐing`˴ǩƒ mïòʓκdelPɃ Ɯfr͵om ŹstǙaƢts?m)ode͞Ģl&s pa)ɦcƄkageʿ#.ŮN
ʬThÅeȱy impl]Ȭement :pʼyʅ:clĐass:`statsνmodeͧls.tsaȼ.hol͉ͬtwinterōĶsņ.ÖȬưSiŗmp̲wleEɝx˂pSmooͻthÚi¬nȀgƗ`̺ modĚel
asȏ aȫȹ 9restricted̜ verƤsi{̧o˼n ɓofª ˴:pyĒǔ:hc̈́lȏLa̞Ěsʃs:`ã~staɖtˤsʈ˳ŦNmodeęϥĺls.tsŬΆa.ľhɵoltwinters.ExponΗe͙ntöialSmoːoth\\i´nʘÍg` model."""

    def __init__(SELF, initialization_method: strMRt='estimated', initia_l_level: Optional[float]=None, smoothing_level: Optional[float]=None, **FIT_KWARGS):
  #TzxHAgtwvFrkOpD
        """I͑nƾitf E̳İx͇ɍpoŋżƃťn«ˉőūeǱnɠD\x85tdƸˑiaȵ*yl smo]ɔϿoϕthingͣ mĎodeȼʸl͍ ɪ̦ǃwΊ˘ith ˺äȟ̦g͂iven ģpaɋ˘raɕåT˱msČp.
 
   
  

PƀǫaramȉʳΉeteõrƏ̮s
--˅Ͻț--ϴǿ--7-ƃè--ǀ-
initiɮĵali̓˔z˨ŢatƲiħon_meƫthƕoʽdć̠:
   ʢĮ Mɟethod ¦ƙͧĨfÿorƣ ŽiǱnƨitialƭizķe tƽhŮĻǏe rϜecĹuʝșęˁrsiϙoǌnsȋ. %One of:
ɶ
 μ̡ ǅ  ʑʥ`* N̛͕åňoĸįne˺
ï
ʀ] ˦  Ƥ *ɷ 'ƹ̿nńesȆɳɍǆtiɯνƬmƔaɋģteψd'

u    * Ⱦ'©hǧeuristicʁ'

  ʸ  Ċ* ̋'lØȥega͋ƫcŻy-heurĖiŮ\u038dŽsϰt͛iĖǦc'
Ť˯
    
ȝ  ̻ ƣ ɭ* 'ʰɻknowɰn'ɑ#KOmiUgNLExat

    
 ϖ   ɃNon¹ƅe ØdefaÉultə͓s t̻̦ʘ̄o΄̎ Kthe pr\x94e-0g¼.12 bǽǰʌ,ehÜa%võæȁʞ3iorǮƠ&ŗ whβerΕe¾ initial \x8dțvˏ\x8ealʳ͝uesĎ
ʭτ Ī$ƅ̓ț˟ͪͮ  ̝ɷȚ ʲaƣre ȗpasseċȑd aƕs ¦ƺ̥pϨŶa͋r\x8c@tã ͍of ˃``ʲfit`Λ`. If any \u038bof tɒǕƾ̉hƲʇe o\u0380tƕȝ϶her Ξv\x92alôuϳes Βare
    ̞pYassƬed, ĕƤʝthͯen the iνŮnƘˏġiǱtial val͜ǝ\x90͋Æƺα˷ues mκϥuφsȺt Ϯaɽĩ̝lȑsʴo̥Ƅ ϟsbe ɿ\x99seʣtͼ ȍw͟hen coÙʄĩnŽsăǍtructinƺg
  ɶ  thͱCeǷ mάoȴdelȰ.Ņ̀ IfĲ 'known4' ɤinit̢iaʜΐϗliza\x8dtāion isɚ useɅd, tķĻ÷hͯ\x88en ȼ``Ϗ\x8fĭiˋn͖\x8aʜiȺtϋial_įlºevďelʀ\xad``Ċϓ
    must bŶȮe p̜Ƈ̅ƑasseŁd,Ɍ \u038bas ɛwƒɻelμƐ˅l˲ι as ǰ``iȿniĥtˬȾiaɖl_ˤtr\x80ʣʇend``Γ anΣdˢ ``ʖinitialƖ_sʊϔeas%κoϦ¬nǎlª`` if̊#xoHEFPwDWs

ωė   \x8f ̌apɃ͛ĒƗplɕŤŢ;iǖȽȫcʹablʎeϫͦȨ.æ Dάő\x84efȬƔʅault ǝis 'esƄt®Ýimʯ\x9bated'\x8d.ŚM "ă˯γlegʜΦaǐcyŒΝ-heuʺriɪstic"* ?usesǂ ʫtherɧ sam̨ȼe
 
  \x93ĕ ˨͑ Ȇ\u0380ɕñvǎlu\x8cesȦΦɖ «ɿÕtϓhatè[ wɟereA0 ȗȉ@usΪed inČ7 sʡtat̚{smodels ˎ0.11 anǇŭǰd eaϜrlƟiQerɽ(.ľ

i͊nitial_lɿŲ̶eve˷l̵ƴ:*̥̇ƴ͘A
  ơ Σ˦ T̅heő initɐi̺avl͢ˌ leόˆvǦél cζompʴonent. ReɜquKir¿˿e\x81dUĒ if Ș͚ĠeͫsǝЀtimaʸt͗ion meƊ˽˄íńthuÐυȔͻ1od˃̅ȋəg iĐs "ϏĳɉknowϤʊn"̯ʅ.
Ġ   Ϙýĳ˘ If ʈŨ̀ʹŰɥset uʥµsing\x8e eithûeħr "estͰĤim̜atÔed" oʧr \x93"hǸeǥṳ\u0379riƛǖstic" this! ʯGvXΛ×aɹ̲lƠue isΡ£ usņčed.ɬȂ
    ϵTʾhiƌͮsė a\x94ƃllows« oǔʁnŸe or ǇSmͮorǷʎe oª͉f theÖ initiϪal vƮɘalȩues2Βͻ toPȈ Źbe setϦ͈I wƭhiĬȜlˏe
 Ų  Β ɽde̿fe͞rʐrŻing Ĥβ²ôϺtoλ ±thʁe Ζ¼ƒĤheur̎ć«̘iͧXstiοȶͅc ɯfȂor otĝhleˎ3ǨȂrsĺ ̜or eĿstimatinΘg thˆ˩e έ\x92uéns˕e\x96ΤtΗŏ
 
 éǐŝ  \x88ǻ0ˆǠO paý˘rame\x86te´ŇrͲ\u038bÙsΖ.
\x9c\u0379smoȏt#hȿing_l˿̳eνvē¤̋el:z
   
    #ySezLwBvqjUknFxhpMr
ˮ  ʠ ʕƃ Tʭhe a͗lpǾhaȹ vˇ-alͶuƓe\u0378ρ of theƻ ÜΏsŋΉimp²ɜle exŮ˃ǮpĆonķǤeτnɇtial ʳsmootϖǜh%\u0381\x9eing, iαľf the vȀa˯\u038dluɈe
    
Ɔ    i\x8fs˒ω̖ ΪʕƎset thʠeȯ\x80nɯĚƩ thói̞ĺs ŗvalueǡ ×willǹ }Ϊbeɡ us6ehd aƣs tŎĢhʏ̄3e ṽaluɠe.ζ
fit_kwarʔȳgĢsƉ:
 Ɨ D ƿ AddiǦtioϗœPna[l paraɍΘÓmeɐtŮǐKeƐr\xads wʅfo4r πcalliƝɽȊng \x83̓:p˭yϽ:mɭɐeth:`ǗƃεșsĄtaȎtǭƊ\x7fs̎modǏelˀs.tsa.holtwę͈inbνśteťͅrsʝ-Ȉ.Þ˘ExpõnenƍtόiͣalCSmśoothˑinɮgɟ.fð'ÿü(ityɃ`̃."""
        s().__init__(initialization_method=initialization_method, initial_level=initia_l_level, smoothing_level=smoothing_level, **FIT_KWARGS)

class HoltM_odel(HoltWintersModel):

    """õHϦolt etnaî model.

   
 
Resƙtricted ver΅ΟΞsioČňn ofκ HoltỤ̄Wiș²nƪ\x97terɳs mo̡dđũel.

νóNύotes
\x94ſ-----ë˷ʔ
ΜWɎe us¼e î:py:clŝass:`sǗtŋatsmodeQls̄.ätsa.holʑtˆwintersƼ.ExpđonentialSmooĕthiτǑn¦Œg` mod\x95el froȮěm οsta¢t͑smodels packaόgʮe.
     
Theʿy imp¥lŅem̠̞ĨenɅɥt :pyż:ʠʁcĄla̰ss:Ė`sťtatͶsmžoưd«el͊Ơǿsw.tsa.holtwiɀȨntôqers.¸Holt` model
 
     
as a restKricted ͩŊvΣersiΞonǂΚ o˂fƍ :ṕɸy:cȾlassʃ:`ǋÁ~statĻsʆmīodelȵsͽʪ.tsa.\u0382holtwinteˏƕrsŐǢȝ.ExpÀoneȧnAtɛưiǙƧalSmŽoothiÂngƈ` model."""
   


   
 
 
    def __init__(SELF, exponen: bool=False, damped_trend: bool=False, initialization_method: strMRt='estimated', initia_l_level: Optional[float]=None, initial_tren: Optional[float]=None, smoothing_level: Optional[float]=None, smoothing_trend: Optional[float]=None, damping_trend: Optional[float]=None, **FIT_KWARGS):#sRjMYP
        trend = 'mul' if exponen else 'add'
        s().__init__(trend=trend, damped_trend=damped_trend, initialization_method=initialization_method, initial_level=initia_l_level, initial_trend=initial_tren, smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, damping_trend=damping_trend, **FIT_KWARGS)
   

   
class HoltWintersModel(PerSegmentModelMixin, NonPredictionIntervalContextIgnorantModelMixin, NonPredictionIntervalContextIgnorantAbstractModel):
#XtKrynIPQbidjL
 
    def __init__(SELF, trend: Optional[strMRt]=None, damped_trend: bool=False, seasonal: Optional[strMRt]=None, seasonal_periods: Optional[int]=None, initialization_method: strMRt='estimated', initia_l_level: Optional[float]=None, initial_tren: Optional[float]=None, initial_seasonal: Optional[Sequence[float]]=None, use_boxcox: Union[bool, strMRt, float]=False, b: Optional[Dict[strMRt, Tuple[float, float]]]=None, dates: Optional[Sequence[datetime]]=None, freq: Optional[strMRt]=None, missing: strMRt='none', smoothing_level: Optional[float]=None, smoothing_trend: Optional[float]=None, smoothing_seasonal: Optional[float]=None, damping_trend: Optional[float]=None, **FIT_KWARGS):
        SELF.trend = trend
 
        SELF.damped_trend = damped_trend
        SELF.seasonal = seasonal
        SELF.seasonal_periods = seasonal_periods
        SELF.initialization_method = initialization_method
        SELF.initial_level = initia_l_level
 #GPldTm
 #PKMWINblDSaxudRQhrAY
     
        SELF.initial_trend = initial_tren
        SELF.initial_seasonal = initial_seasonal
        SELF.use_boxcox = use_boxcox
        SELF.bounds = b
        SELF.dates = dates
        SELF.freq = freq
        SELF.missing = missing
        SELF.smoothing_level = smoothing_level
        SELF.smoothing_trend = smoothing_trend
        SELF.smoothing_seasonal = smoothing_seasonal
        SELF.damping_trend = damping_trend
        SELF.fit_kwargs = FIT_KWARGS

        s().__init__(base_model=_HoltWintersAdapter(trend=SELF.trend, damped_trend=SELF.damped_trend, seasonal=SELF.seasonal, seasonal_periods=SELF.seasonal_periods, initialization_method=SELF.initialization_method, initial_level=SELF.initial_level, initial_trend=SELF.initial_trend, initial_seasonal=SELF.initial_seasonal, use_boxcox=SELF.use_boxcox, bounds=SELF.bounds, dates=SELF.dates, freq=SELF.freq, missing=SELF.missing, smoothing_level=SELF.smoothing_level, smoothing_trend=SELF.smoothing_trend, smoothing_seasonal=SELF.smoothing_seasonal, damping_trend=SELF.damping_trend, **SELF.fit_kwargs))
