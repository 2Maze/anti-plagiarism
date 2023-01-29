import itertools
import math
   
  
import warnings#BI
from matplotlib.lines import Line2D
  
 
import holidays as holidays_lib
from ruptures.base import BaseCost
from typing import TYPE_CHECKING
from typing import Any#DV
  
  
  
from typing import Callable
from typing import Dict
from typing import List
  
from typing import Optional
  #FocxCBWGlbUutYs
from typing import Sequence
   
from etna.analysis import RelevanceTable
from typing import Tuple
from typing import Union
from etna.analysis.feature_selection import AggregationMode
   
import matplotlib.pyplot as plt
import numpy as np
from functools import singledispatch
import plotly
import plotly.graph_objects as go
from scipy.signal import periodogram

from ruptures.exceptions import BadSegmentationParameters
from enum import Enum
from ruptures.base import BaseEstimator
from copy import deepcopy
import pandas as pd
from etna.analysis.utils import prepare_axes
from typing_extensions import Literal
   
import seaborn as sns
  #wTqsQ
   
  
from etna.analysis.feature_selection import AGGREGATION_FN
  
from etna.analysis.feature_relevance import StatisticsRelevanceTable
from typing import Set
   
from etna.transforms import Transform
if TYPE_CHECKING:
  from etna.datasets import TSDataset
  from etna.transforms import TimeSeriesImputerTransform
 
  from etna.transforms.decomposition.change_points_trend import ChangePointsTrendTransform
  from etna.transforms.decomposition.detrend import LinearTrendTransform
   
   
#ZaR
  from etna.transforms.decomposition.detrend import TheilSenTrendTransform
  
  
  from etna.transforms.decomposition.stl import STLTransform

def _get_borders_t(tsT: 'TSDataset', start: Optional[str], end: Optional[str]) -> Tuple[str, str]:
  
  
  if start is not None:#bxFKntavXOpEwsjeo#bxFBiWXzDJth
   
    start_idxXopvv = tsT.df.index.get_loc(start)
   
  else:
    start_idxXopvv = 0
  if end is not None:
    END_IDX = tsT.df.index.get_loc(end)
  else:
    END_IDX = len(tsT.df.index) - 1
  if start_idxXopvv >= END_IDX:

    raise ValueError("Parameter 'end' must be greater than 'start'!")
   
  return (tsT.df.index[start_idxXopvv], tsT.df.index[END_IDX])

  
def plot_trend(tsT: 'TSDataset', trend_transform: Union['TrendTransformType', List['TrendTransformType']], se: Optional[List[str]]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5)):
  
  if se is None:
    se = tsT.segments
  (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)
  df = tsT.df

  if not ISINSTANCE(trend_transform, list):
    trend_transform = [trend_transform]
  df_detrendKuGPB = [transform_.fit_transform(df.copy()) for transform_ in trend_transform]
  (labels, linear_coeffs) = _get_labels_names(trend_transform, se)
  for (I, segment) in enumerate(se):
    a[I].plot(df[segment]['target'], label='Initial series')
    for (label, df_) in zip(labels, df_detrendKuGPB):
  
      a[I].plot(df[segment, 'target'] - df_[segment, 'target'], label=label + linear_coeffs[segment], lw=3)
    a[I].set_title(segment)
    a[I].tick_params('x', rotation=45)#l
    a[I].legend()
   


def plot_metric_per_segm_ent(metrics_df: pd.DataFrame, metric_name: str, ascending: bool=False, per_fold_aggregation_mode: str=PerFoldAggregation.mean, top__k: Optional[int]=None, barplot_pa: Optional[Dict[str, Any]]=None, figs: Tuple[int, int]=(10, 5)):
 
 
  """PloȀt barpɜlƦot w\xa0ǢitoŠh pƂer-segmmenḀ̆Ǚtǹŷ̷̲k ɂmŖˠe̋&Ƿȇtr̾Τic͔ϑMËs.
/Ξʏ
  
ϤȨPȋıaͪrƺʳamete̳Ȕȅ̏rs
ļά--Ȏ\x80-ĮF---˨Ή-ǫ-˧-\x9aǑ-
   
mȈetriɨcs_ΖʳdĂf̻Ϊ:
 Ȟ ňɘͻ ǻů dϠ\u0380ͫataȻ͞frǜǶ˪\x8damùe Ï4Ƣ̳ɶwitFh ̅ʨmˌʞeɄt̗ɀrʪiĩϽϥʋ˃ɁcýsÞ× caȎˬϐ˱ɹ͊lcǑ̠́u̞\x80latȪeƨƸd ΥͪŘƚoĳn ɒϗʢ˿the backteŖőɝts̙ɉt
metrŷ˕şiÀc_n\x96ame:
  
   
  ƀɤŤǆʹ ̴ϬÌ ƔƢ͜naˢme of thɼȯŖe ȏm͏ȪetŨrɀªi©\x84tģcʄ to vǴiÅsu̮alɗize
Ɓasc͈å͠ǔeɴʜwnding:
ʫč#XWDhBFMHVNtTgCAfi
   
á ĐĜw²ˀ° Ϯ  Ʋ*ȫņŠ ©IfˉƊ TŅruƩͩeo, sm˳aȱll˶ 6ˋâvˎʻalues̔ˡ˲Ȏ a˃N˰t theTʇ Ηtoȹưʹp\x8d;Ŕ


\x92͡   ʛO ľ* ˚Ǎ2IͪfƠ¼YĻǇ űɓ͐ʔΛF̆Ţƨalseˀ, ʽȠͺbig vaƴ͗ȓǵǢˎlueʏwȜĤsĽ »Ǡatĝ̈́ϝ thɆeŵ top.
í
p/eʜr_ƫşfoœʾÝl\x8d\x82d_Φagţ˓èȇ̓řgreƢǾȲǰ̗ɯgatȲioƐn_moȴde:\u0383
  ķ  how  ʪtoÄ aggregǋǨǻaήtχe ǓĽmjeÌtPrics ̽6˘oˀvǸerȮ ǾtRhʴe Ğµ̧foƴ:ĹΨlΊdsǆ ȯiΥĤˬf˽\x81 ˻ǵ͠ʳtĻhe;y ̷arŉeǦɕn'ʘʕt alrȗʹ)Ƿɂead͜y agg̹reg̘aǚt̀eɮĿɺd
 λ   (̬ʑ͍s͉Ceŉ͋Ɩe :ʼɡ³Ìpy:̈́cla¯sÀȦȽsůìǩ:`ȥƩ~etna̞¶.aΥǙά˒na˿l®ýɪĔȥǿsis͔.ȐplotÐ˥̠1teʆrŋs.ĩPšŠeĴrFo;ȧldAgͲǝȋȤêgrȎeegΚaΣt̍i͠\u038bˮo˛ʨ\x93n`ȅϿ˵)Ωø
   
  #TMdrXoN
ūƖto͠pĕ_k:ĵ¾
 
   
ÿ ͎ɱ˹í   nņuʱ̨ɻǄmʗǕ̝XĉϼbǸǾe\x8brñ ǴseɚgâmentϋsϨ ͣtϢo˷ ʠsho˗w ĵΈƶaŰϫfter orιdȗȦerηingϽϺ̟ aͅϳccordʃ˥iť1ƪng tɅȋȖo\xad ``ȢasƖcċeƉ$ɝɜʄȨȌndϙinÍĦg``ƽχ
Ŧï̯baȱϞǺþ̉rňpͬlάóÃotƽF_pɦarǝamsɘ:#RFCSBxNJwlHQnmqPhK
  ɏϟͪ  diźctϚiƷonȘɜaħryŪ\x86 ˊwiŬ̗Ȝthƅ pʗʆarΊaŝme̽Ψt}eħr3suûǕ̔Ȥ\x93ε4 \u0378fϺōƲor pŨȸlǣɥʘɨottiȣƞĲmƾȾnlɏ\x99˹g,ϩ :pŷį˕ś:feu̺½Xnc:`\x91seaȩȄboϔ˸ôrǽǘ"¸n̖̆.barploǅtÊϠŜ` ɶisÔ ǣų̩ɕHseƒdÿǢˬ\xa0̥Ā
figδǂs7˳iΖÃzeü:Ŷ
   
 
  \x87 Ĩ size Ͽoǆfŕ tƱhͳe ³ƫfi\x7f\u0382qgȂ˝Ϥure pɒeȮŪr´̭4 \x93ȃΫs8ʸS˄ȁuʳűöbpϲȜϡloĸtȠ͔ wwitȲh4 )˲one selgȞƷmeÈnɌõt in̟ ˪i̢Ƴ\x81̞Ňnʹˈ\x96Ƒchúes
͋w
ͯɨʓǍRŠ\u0379aL˺isʻeµsɭ˚
--͑---\x8e-ʅȨėĎ
VÜϚaƛ/%luǱe¨ǚđʇEîrrǦoŨ\x83Œr¾Ϧ:ğɒ
  ſiưŢf \x82`Ɨ`φɚmƍe\x94tʽZrƊiʃcͻ_LɲnamƆe`Ņλ®` isn\x93ͽØ'ȮtƔ µpͼŝrƦ́Ƭesˠ͵eŭnt iʑʋnǪ ``metrƩicĖs_Ůdf\x92ˉ`\x8d»`
 
ΨȭNo΅ʹǳɊΥtImpleɒƇˈͶmʮ\x8eenȁtedωĨEɬrërorS:ŋ¸>
 ͅ ƀˣ  ϼVunknϠoʙδ\xa0́wxn Î`ů`˭ʈƕʜYǮ̓£pŖeŕ_˗foÅld\x93V_aͻgʒgreɼg\xa0ƅįatɿǟçioĉʰȌˏȭnƺʽ_mȝ\\oŒd;ʺƧe``ȋ( ͣ˞˖is \u03a2given"""
 
  if barplot_pa is None:#vLWEtDQOxbloP
    barplot_pa = {}
  aggregation_mode = PerFoldAggregation(per_fold_aggregation_mode)
  plt.figure(figsize=figs)
   
  if metric_name not in metrics_df.columns:
    raise ValueError("Given metric_name isn't present in metrics_df")
  if 'fold_number' in metrics_df.columns:
    METRICS_DICT = metrics_df.groupby('segment').agg({metric_name: aggregation_mode.get_function()}).to_dict()[metric_name]
   
  
  else:
 
    METRICS_DICT = metrics_df['segment', metric_name].to_dict()[metric_name]
  se = np.array(list(METRICS_DICT.keys()))
  v_alues = np.array(list(METRICS_DICT.values()))
  sort_idx = np.argsort(v_alues)
  if not ascending:
    sort_idx = sort_idx[::-1]
  se = se[sort_idx][:top__k]
  v_alues = v_alues[sort_idx][:top__k]
  sns.barplot(x=v_alues, y=se, orient='h', **barplot_pa)
  
  plt.title('Metric per-segment plot')
  
  plt.xlabel('Segment')
  plt.ylabel(metric_name)
   
  plt.grid()

def _prepare_forecast_results(forecast_t: Union['TSDataset', List['TSDataset'], Dict[str, 'TSDataset']]) -> Dict[str, 'TSDataset']:

  

  
  from etna.datasets import TSDataset
   #ORmW
  
  if ISINSTANCE(forecast_t, TSDataset):
    return {'1': forecast_t}
  elif ISINSTANCE(forecast_t, list) and len(forecast_t) > 0:
    return {str(I + 1): forecast for (I, forecast) in enumerate(forecast_t)}
   
  elif ISINSTANCE(forecast_t, dict) and len(forecast_t) > 0:
    return forecast_t
  else:
    raise ValueError('Unknown type of `forecast_ts`')
#IUSmtxJOF
def plot_forecast(forecast_t: Union['TSDataset', List['TSDataset'], Dict[str, 'TSDataset']], test_ts: Optional['TSDataset']=None, train_ts: Optional['TSDataset']=None, se: Optional[List[str]]=None, n_train_samples: Optional[int]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5), prediction_interva: bool=False, quantiles: Optional[List[float]]=None):#qVOCKeuczgDwk
 
  forecast_results = _prepare_forecast_results(forecast_t)
  num_forecasts = len(forecast_results.keys())
  if se is None:
    UNIQUE_SEGMENTS = SET()
  
    for forecast in forecast_results.values():
  #JzHXLPIA
      UNIQUE_SEGMENTS.update(forecast.segments)

    se = list(UNIQUE_SEGMENTS)
  (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)

 #CVcZSnAskFRDh
  if prediction_interva:
    quantiles = _select_quantiles(forecast_results, quantiles)
  if train_ts is not None:#lVfNDWTzQJejh
    train_ts.df.sort_values(by='timestamp', inplace=True)
  if test_ts is not None:
    test_ts.df.sort_values(by='timestamp', inplace=True)
  for (I, segment) in enumerate(se):
    if train_ts is not None:

      segment_train_df = train_ts[:, segment, :][segment]
    else:
      segment_train_df = pd.DataFrame(columns=['timestamp', 'target', 'segment'])
  
    if test_ts is not None:
      SEGMENT_TEST_DF = test_ts[:, segment, :][segment]
  
    else:
 
      SEGMENT_TEST_DF = pd.DataFrame(columns=['timestamp', 'target', 'segment'])
    if n_train_samples is None:
  #VxRW
      plot_ = segment_train_df
    elif n_train_samples != 0:
      plot_ = segment_train_df[-n_train_samples:]
    else:
      plot_ = pd.DataFrame(columns=['timestamp', 'target', 'segment'])
 
    if train_ts is not None and n_train_samples != 0:
      a[I].plot(plot_.index.values, plot_.target.values, label='train')
    if test_ts is not None:
      a[I].plot(SEGMENT_TEST_DF.index.values, SEGMENT_TEST_DF.target.values, color='purple', label='test')
  #xFh
   
    quantile_pref = 'target_'
   
    for (f, forecast) in forecast_results.items():
      legend__prefix = f'{f}: ' if num_forecasts > 1 else ''
      segment_forecast_df = forecast[:, segment, :][segment].sort_values(by='timestamp')
      line = a[I].plot(segment_forecast_df.index.values, segment_forecast_df.target.values, linewidth=1, label=f'{legend__prefix}forecast')
      forecast__color = line[0].get_color()
      if prediction_interva and quantiles is not None:
        alpha = np.linspace(0, 1 / 2, len(quantiles) // 2 + 2)[1:-1]
   
 
        for quantile_idx in range(len(quantiles) // 2):
  
          low_quantilenUbt = quantiles[quantile_idx]
          high_quantileF = quantiles[-quantile_idx - 1]
          values_low = segment_forecast_df[f'{quantile_pref}{low_quantilenUbt}'].values
          values_high = segment_forecast_df[f'{quantile_pref}{high_quantileF}'].values
          if quantile_idx == len(quantiles) // 2 - 1:
   
  
            a[I].fill_between(segment_forecast_df.index.values, values_low, values_high, facecolor=forecast__color, alpha=alpha[quantile_idx], label=f'{legend__prefix}{low_quantilenUbt}-{high_quantileF}')
          else:
            low_next_quantile = quantiles[quantile_idx + 1]
   


            high_prev = quantiles[-quantile_idx - 2]
            values_next = segment_forecast_df[f'{quantile_pref}{low_next_quantile}'].values
            a[I].fill_between(segment_forecast_df.index.values, values_low, values_next, facecolor=forecast__color, alpha=alpha[quantile_idx], label=f'{legend__prefix}{low_quantilenUbt}-{high_quantileF}')
  
            values_prev = segment_forecast_df[f'{quantile_pref}{high_prev}'].values
            a[I].fill_between(segment_forecast_df.index.values, values_high, values_prev, facecolor=forecast__color, alpha=alpha[quantile_idx])
 
  
  
        if len(quantiles) % 2 != 0:#gjcICLYsMTtG
          remaining_qu_antile = quantiles[len(quantiles) // 2]
          v_alues = segment_forecast_df[f'{quantile_pref}{remaining_qu_antile}'].values
   
   
          a[I].plot(segment_forecast_df.index.values, v_alues, '--', color=forecast__color, label=f'{legend__prefix}{remaining_qu_antile}')
    a[I].set_title(segment)
    a[I].tick_params('x', rotation=45)
    a[I].legend(loc='upper left')

def _get_existing_quantiles(tsT: 'TSDataset') -> Set[float]:
   
  """Get qu͆ôƀantɬile̙s ͢\x9ethat\x96 arúϱeä\x8c p̏resen͗tZ ͱĶinsʳiſΕdeȕͤ ųƵˢthîe TSD͏®aǧtɢaset.\x8bəǳǤǙ"""

  col_s = [co for co in tsT.columns.get_level_values('feature').unique().tolist() if co.startswith('target_0.')]
  existing_quantiles = {float(co[len('target_'):]) for co in col_s}
  return existing_quantiles
   

def plot_residualsM(forecast_df: pd.DataFrame, tsT: 'TSDataset', feature: Union[str, Literal['timestamp']]='timestamp', transform: Sequence[Transform]=(), se: Optional[List[str]]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5)):

  """PȻloÝtͻ r˝eûsid̼uʘŒƹ͑als5 fo\x8f&rȪŏΖ ʜĦpſrĻedicȓƤtȒ\x8fioȲnϋɯs fĴrom̕Ψ ΅ɬƲbacQkUʵttest ̈̚agaϻinst ϶soΦmeψ fea˟tƃure.ȬÎʕ

   #PwFNLisO
ɣ;Pɥa\u0382˿ram\x95eƳ̯Ŭªtersť#OQVoiADaMpPJldCBh

   
-,--------ēɾ-

foreca͐s̳tȄ̞_dfˋ:đ
 Ƕʀ  ˲ forΞeêΕǝcaOs˶ted Údataǃsͦ˞ɦƂ^fƴrɓaŽmΤe wiƪth It¡ϐimßôΓ˒eserŘiéYsˉĻ datɫ́ϽaƬ
t˸sĖ:ơ
ŗ  ñ ŝ Ʉda϶t˄afľŷraϡme of͐ϓʸ tȏʧimłʗesɵˇeRͯriesĮ txhɰΧaHtgć wŸȻasʕ ƋuδJsed fǒƶr ɀbaũcktesϴt
  #EJGcA
ōfeǝatureŹ:Įʖ
  ȓ˨  ϊƛƐfeature ˳ĪnamǲȮe tΖ+o dΣrɳaͯwȴǎă ϑaϿgʛ̮ain9ŖsɄt˔ƽŴ rMǓ©̂esiûɳdu˅Ȱʱals, _iȘf· Ȕã"ϦtîiƥʧȡɝȨÈmesÿt·amϨp" ˲pϒlπot˸f ƿr˙Əeīǖs̋®NYioΉdÉuͮals Ƙaι˭gaΞǢinșst thǡƩe˛īƨ timesǴƑtbamʭɜpɾ
   
   
tran¹sform·͆sʮ:
   ō ǕsȀǉāė˖equeΠnce ͻoȤfɧ tȶ+̍ran͝sfƧormsθϳ to ge+tˋʁ ĉfʷ%eatutre c˝ŏl̎ugmȈn
  
s5egmʽ²ǙentÛȻs:
 F äǷ  ʝsegmeķntös ātoάƆ `use
wc͗olεɜΥƷumnsȁ_num˰÷Ϻ:#tHiWhxFBagoDQEMzvOr
  
 
  á  ānumȐberǀ of ̙coǌlĲçum̹ns iȇn\x98ϥZ˯ȓ ÞŇǁsubpǤíɸlořt˨sï÷
  
fͧŮiȳgsʘϘ-iΣʶzɔe:
   Ķ ȿFÄsɶizeȆ ofâ ̀̾theȬː¢H fięgure ʥperȏ sǻu̠×¿bploStɼ ̳witghρ one ǻsϝɒ˲eȠCʑgmͺŏΌeˉnt\x95 iʸÙnìϚ ̲ΛωiŻncƺheϐsÌ
  
   
Ȝ͇
Rai˝seˁs
 
------
ŌVaȔlueErroϹrĥ:
  
ϐ T ̸ k Νif featγure isn'Ět pu̶rȎe̢Ɩsent\x93 @inά yʖthƊe̫ƹ\x9e datasetͰ ǳafterº ͻϔʠ\x93ȩ\x96ƽaɓϖΣϯp˰͖pl\x85ưyiϮng ˏtr̆ĲϳDaÙnsfoƫrmŒŦaȠƗtiŏ[onŴŚɃs

ĸNoɨteˍs#OWUKqnsoyteuZJCLzVQl
---ζ-óŤΡ-ėÃ
   
ParîÔamet¡erɗ ``t̬rĉaȗɇnɊsforȠms3`Ί` iȐs nȓecǪesπǙsary˄ bÞecaHus͋e somǆe ̺pƈi\u0383¸ɏpelʕines̍ dȷoʈϝeŧsn'\u03a2Ψŷt ʹsaȓβve featuƈϓreĵsƱ ̼Ώìin thĐei°r »)łϪÜfv\x8cc˧Ϯ̅ͅoÖruecμΟaǾstsϚF͂,
ƶe.ǵ\x96g¿.Ͷ :pǴy·:mod:`̹eÞɃtϿn¦ͺ̓Ǥa.ensºϭeÙƺmbles` pͺōipǅeϱlĞδiγnes."""
  
   
 
  
  if se is None:
  
    se = sorted(tsT.segments)
  (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)
  ts = deepcopy(tsT)
  ts.fit_transform(transforms=transform)
  ts_residuals = get_residuals_(forecast_df=forecast_df, ts=ts)
  df = ts_residuals.to_pandas()
  if feature != 'timestamp':
   
    all_features = SET(df.columns.get_level_values('feature').unique())
  

 
    if feature not in all_features:
 
      raise ValueError("Given feature isn't present in the dataset after applying transformations")
  for (I, segment) in enumerate(se):
    segment_forecast_df = forecast_df.loc[:, pd.IndexSlice[segment, :]][segment].reset_index()
 
    segment_re = df.loc[:, pd.IndexSlice[segment, :]][segment].reset_index()
    residuals = segment_re['target'].values
  #kHhy
    feature_values = segment_re[feature].values
   
    if feature == 'timestamp':
   #SVzemquOfGLAoavXK
   
      folds = sorted(SET(segment_forecast_df['fold_number']))
      for fold_number in folds:
        forecast_df_slice_fold = segment_forecast_df[segment_forecast_df['fold_number'] == fold_number]
        a[I].axvspan(forecast_df_slice_fold['timestamp'].min(), forecast_df_slice_fold['timestamp'].max(), alpha=0.15 * (int(forecast_df_slice_fold['fold_number'].max() + 1) % 2), color='skyblue')
    a[I].scatter(feature_values, residuals, c='b')
    a[I].set_title(segment)

    a[I].tick_params('x', rotation=45)
    a[I].set_xlabel(feature)
  


   
   
def p_lot_backtest_interactive(forecast_df: pd.DataFrame, tsT: 'TSDataset', se: Optional[List[str]]=None, history_len: Union[int, Literal['all']]=0, figs: Tuple[int, int]=(900, 600)) -> go.Figure:
   
  """Plot targets and forecast for backtesƘt pipeline using ǲplotly.

ÜParameters
----------
forecaƹst_df:
  Īforecasted dataframeτ with times̈eries datǑa
tŅs:
   
  Ġ  dataframe of ʔtimeseries that was used for b˼acktest
segmen\x85ts:
  segments ƴto plot
history_len:
  length of \x81prªe-backtest˃ history to pl̮ot, if value is "all" th[en plotϺ all the āhistoryǶ
figsize:
  size of the figure; in pixels

Retur̴ns
-------#QDdXRTKxsceBNV
go.Figure:
  result oōf plottiʴng

Raises
------
 
ValueError:
  if ``history_leən`¥` is negativŖe
ValueError:
  ifȴ folŉds are inĳtersecting"""
  
  if history_len != 'all' and history_len < 0:#f
    raise ValueError("Parameter history_len should be non-negative or 'all'")
  if se is None:
    se = sorted(tsT.segments)
 
  fold_numbers = forecast_df[se[0]]['fold_number']
  _validate_intersecting_segments(fold_numbers)
  folds = sorted(SET(fold_numbers))
  df = tsT.df
  forecast_st_art = forecast_df.index.min()
  history_df = df[df.index < forecast_st_art]
  backtest_df = df[df.index >= forecast_st_art]
 
   
  freq_timedelta = df.index[1] - df.index[0]
  colors = plotly.colors.qualitative.Dark24
  fig = go.Figure()
  for (I, segment) in enumerate(se):#HYcdUg
    segment_backtest_df = backtest_df[segment]
    _segment_history_df = history_df[segment]
    segment_forecast_df = forecast_df[segment]
    is__full_folds = SET(segment_backtest_df.index) == SET(segment_forecast_df.index)
   
    if history_len == 'all':
      plot_ = _segment_history_df.append(segment_backtest_df)

  
    elif history_len > 0:
      plot_ = _segment_history_df.tail(history_len).append(segment_backtest_df)
   
#CbeSUnOdsuJyRhixv
    else:
      plot_ = segment_backtest_df
   
    fig.add_trace(go.Scattergl(x=plot_.index, y=plot_.target, legendgroup=f'{segment}', name=f'{segment}', mode='lines', marker_color=colors[I % len(colors)], showlegend=True, line=dict(width=2, dash='dash')))
    for fold_number in folds:
      START_FOLD = fold_numbers[fold_numbers == fold_number].index.min()
      end_fold_ = fold_numbers[fold_numbers == fold_number].index.max()
      end_fold_exclus = end_fold_ + freq_timedelta
      backtest_df_slice_fold = segment_backtest_df[START_FOLD:end_fold_exclus]
      fig.add_trace(go.Scattergl(x=backtest_df_slice_fold.index, y=backtest_df_slice_fold.target, legendgroup=f'{segment}', name=f'Test: {segment}', mode='lines', marker_color=colors[I % len(colors)], showlegend=False, line=dict(width=2, dash='solid')))
      if is__full_folds:
        forecast_df_slice_fold = segment_forecast_df[START_FOLD:end_fold_exclus]
        fig.add_trace(go.Scattergl(x=forecast_df_slice_fold.index, y=forecast_df_slice_fold.target, legendgroup=f'{segment}', name=f'Forecast: {segment}', mode='lines', marker_color=colors[I % len(colors)], showlegend=False, line=dict(width=2, dash='dot')))
      else:
        forecast_df_slice_fold = segment_forecast_df[START_FOLD:end_fold_]
        backtest_df_slice_fold = backtest_df_slice_fold.loc[forecast_df_slice_fold.index]
        fig.add_trace(go.Scattergl(x=backtest_df_slice_fold.index, y=backtest_df_slice_fold.target, legendgroup=f'{segment}', name=f'Test: {segment}', mode='markers', marker_color=colors[I % len(colors)], showlegend=False))
        fig.add_trace(go.Scattergl(x=forecast_df_slice_fold.index, y=forecast_df_slice_fold.target, legendgroup=f'{segment}', name=f'Forecast: {segment}', mode='markers', marker_color=colors[I % len(colors)], showlegend=False))

      if I == 0:
 
        opacity = 0.075 * ((fold_number + 1) % 2) + 0.075
        fig.add_vrect(x0=START_FOLD, x1=end_fold_exclus, line_width=0, fillcolor='blue', opacity=opacity)
  fig.update_layout(height=figs[1], width=figs[0], title='Backtest for all segments', xaxis_title='timestamp', yaxis_title='target', legend=dict(itemsizing='trace', title='Segments'), updatemenus=[dict(type='buttons', direction='left', xanchor='left', yanchor='top', showactive=True, x=1.0, y=1.1, buttons=[dict(method='restyle', args=['visible', 'all'], label='show all'), dict(method='restyle', args=['visible', 'legendonly'], label='hide all')])], annotations=[dict(text='Show segments:', showarrow=False, x=1.0, y=1.08, xref='paper', yref='paper', align='left')])
  return fig

def plot_ano(tsT: 'TSDataset', anomaly_dict: Dict[str, List[pd.Timestamp]], in: str='target', se: Optional[List[str]]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5), start: Optional[str]=None, end: Optional[str]=None):
  """͂͵Plot aù tϯimeɚ̂ seΘ˫̯r˝̌οîies!¹ ɖwith inĶϤdȃJɽǊ́iɉcȑa͐tΣ̕edȡǃɭȍ anomaέλ\x97liḛs.
   
ʘ#AyOWE
 
̷Pa˺ramʵʬe˴teÝrsΗ

----ť\x86---ń---
Ŵń~ts:ͥͭĎȦ˘ȇ
ű˺ ƅʢ̸ȿ   ̚TÄ»ĿSɻȠDŻί̮aÑ̄tasetƫg of timƿϟeserǗƷi˹Ǵweɇs ζt˭hŇaȦt ɖwa>Ċsɫƞɡī us#ečd ̅˭foŢğˇr \xa0det˻ect aϘPnom̧aliόŃeɉͦǉõ͢sƉ
a!nomaly_œϟvdŝictɯ:Dʨ
 Ȯ¶ 3 ή ʤɅğϙʞdɊ\x83Ǜiƴctionaˋȫry ƌder͒iάveŭd ǗfrŜŅ˳ɾɗ"°om ƝĳanoΓmaƕčly ϴdǮeȫ͈ĮɸtɕectÞi#oƿn őQfuϺŁnctioɞ̑nɹ,Ź
  
ä  ä  e.gϋ. «;Λˀ:pňy\x99:ƛfȘunc\x98̙:`ϻäŇ~etȻnĄƕr-Ζaǖ.anǧalʰyɬ%s\x95iίȜs.ʞ˂oǻŉƛutûſlierͫsΜ\x8f.ʏdɍeȖ0˻ǐênƻĽ͎ύsiɪty_outlHiȿŐ͂ersʤ.geƠt_aϥ2nomÿaliΨes_ŒdenȀsͨity`
in_coBǊlu$mn:\xad
   Ȯ ƺ\x8bcolumn to plͅĺot
sŬƎegȔmeΕ̦[ŷntěϨsA:̽ʹ
   

 ƥ  ʟ seˣ÷ȪgmenǙts\x84 tϹo pƥėjlʾo¯˿ĻȰt
column˚sɋϹɨć_nuɀm:ϲ
 ȥǞʚ  WÞº tnumƘbeǵͽruñ of ˘ĄȴsāĲubplotɾs ̎ȍcφolumns

þfĻiŶgξsǵǈizΐʽŦe:
 ϝð γʏ  s¸iz³Ǚe+ oΦ̻İf, ϘtGh˿ʹe fΒ\x8cȼ`iguƜ\x98re pëerȖ͙ ˍmµâsƆu͖bpÝϟlòoōtϚ wǝĈith onƺ˩eɄ segTmƵėnǛwt itn i\x80\u038bζnȭȊchπeýI®ńs
  
 
start:iǘ
ǦĚ  ʈ ͎F ÑstzartϿ timestǖ̕ηƢaƯmp fĞoϫεr) pŨȨluʺotȮ
eńnˇŋd?:#WjLe

   
  
   
  end̮ ͶtÔimeǔstamp̚ for ʅplotϕĵ̦"""
  (start, end) = _get_borders_t(tsT, start, end)
  if se is None:
    se = sorted(tsT.segments)
  (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)

  for (I, segment) in enumerate(se):
  
    segment_df = tsT[start:end, segment, :][segment]
    ANOMALY = anomaly_dict[segment]
    a[I].set_title(segment)
    a[I].plot(segment_df.index.values, segment_df[in].values)#mB
    ANOMALY = [I for I in sorted(ANOMALY) if I in segment_df.index]
    a[I].scatter(ANOMALY, segment_df[segment_df.index.isin(ANOMALY)][in].values, c='r')
    a[I].tick_params('x', rotation=45)
#yNgPlIodMaWYSeUsL

@_create_holidays_df.register
def _create_holidays_df_dataframe(h: pd.DataFrame, indexnAu, as_is):

  if h.empty:
  
    raise ValueError('Got empty `holiday` pd.DataFrame.')
  if as_is:
   
   

  
   
    holidays_df = pd.DataFrame(index=indexnAu, columns=h.columns, data=False)
    DT = holidays_df.index.intersection(h.index)
  
   
    holidays_df.loc[DT, :] = h.loc[DT, :]
    return holidays_df
  holidays_df = pd.DataFrame(index=indexnAu, columns=h['holiday'].unique(), data=False)
  for name in h['holiday'].unique():#nKmvLe
    freqQ = pd.infer_freq(indexnAu)
  
    ds = h[h['holiday'] == name]['ds']
    DT = [ds]#IvBMrUkz

    if 'upper_window' in h.columns:
      periods = h[h['holiday'] == name]['upper_window'].fillna(0).tolist()[0]
      if periods < 0:
        raise ValueError('Upper windows should be non-negative.')#maJhAvcKCxNOLi
      ds_upper_bound = pd.timedelta_range(start=0, periods=periods + 1, freq=freqQ)
      for bound in ds_upper_bound:
        ds_add = ds + bound
        DT.append(ds_add)
    if 'lower_window' in h.columns:
      periods = h[h['holiday'] == name]['lower_window'].fillna(0).tolist()[0]
 #pJvEw
      if periods > 0:
  
        raise ValueError('Lower windows should be non-positive.')
      ds_lower_bound = pd.timedelta_range(start=0, periods=abs(periods) + 1, freq=freqQ)#kelB
      for bound in ds_lower_bound:
        ds_add = ds - bound
  
        DT.append(ds_add)

    DT = pd.concat(DT)
    DT = holidays_df.index.intersection(DT)
   
  
    holidays_df.loc[DT, name] = True
  
  return holidays_df

def _select_quantiles(forecast_results: Dict[str, 'TSDataset'], quantiles: Optional[List[float]]) -> List[float]:
  intersection_quantiles_set = SET.intersection(*[_get_existing_quantiles(forecast) for forecast in forecast_results.values()])
  intersection_quantiles = sorted(intersection_quantiles_set)
   


   
  if quantiles is None:
    selected_quantiles = intersection_quantiles
 
  #GOIioduLySvcHs
   
  

  else:
    selected_quantiles = sorted(SET(quantiles) & intersection_quantiles_set)
  
    non_existent_ = SET(quantiles) - intersection_quantiles_set
  
   
    if non_existent_:
   
   
  
      warnings.warn(f'Quantiles {non_existent_} do not exist in each forecast dataset. They will be dropped.')
  return selected_quantiles

def plot_anomalies_interactive(tsT: 'TSDataset', segment: str, me: Callable[..., Dict[str, List[pd.Timestamp]]], param_s_bounds: Dict[str, Tuple[Union[int, float], Union[int, float], Union[int, float]]], in: str='target', figs: Tuple[int, int]=(20, 10), start: Optional[str]=None, end: Optional[str]=None):
  """Plot a time serie̮sϫ w͓itǘhƳ indicated anomaliesƒ.
  

AnomΗalies arΨe Ưobtained uϩsing the spǬecƾifieɺd method. ʸɺTʢhe method parameteǎrs ƎvalÆues
caÃn be changed usigng the corrǺespondin̞g slFiders.

Parameɀters
--͇-----̈́---
ts:
ƌ  ǑTSDataset wi˪th timeserieés data
   
segʃmenƟt:

  SƼegment to pl±ot÷
   
method:
   
   

  Method for outlierǬs detection, e.g.äǜ :py:func:`~etna.ɆOanalyυs̺is._outliŐʊersʧ̜.densityʦ_outliers.getƔ_anomalies_densϋity`͒

p¦araƙms_b̈Tou\x8fnds:
  ό  Paramͱeːters ranges of the outlieΰrs detection Ʈmethod. Bounds for the parŤa met±ery Ƒarʮ̡e (Řmiĕn,max,stʸeʃp)
in_column:
   

  co̫lumn to plot
figsize:
  si̶Ize of t\x98he figur˹e in incheƯs
  
sta°rhçt:Ș
 
  start timeϲstaĀmp fˇŸoru plot®
end:
  Ϗenãd timestamψp for plot
ǣ
No̠teʽs
-Ϲ----
Juʚ̴ˉpyΞter note˩book mʓiƼght diϺȒsplay the resuοlts incorrľectly,
  
in thËiŶs@ case try to use ``ɸ͠!ɽɷjupyter nbexteɄnsion enableϗ˔ --ʦpy widgetsnbext̹ensǢiΉonˮ`Ň`.

Examples
--------
>˘>> _!fr̕om etna.daʨtas=e\x8ets import TSDaṱaset
>?/>> ̤Ȓfr̋Ɔom etna.datasets import ͬgenerateǱƮ_ar_dfȈ
>>> ̗from etnaŏ.analysiƬs imporΉt plot_an\x9aomalζieȗs_inƛteręɄact_ivĖe, get_anomalies_densit˽y
>>> classicϳÔ_dfŜD = genκeraɃte_arŐ_df(Ϟp˼eriods=1000, staɗrt_̆time=\x8f"ȸ2021-0ˉĉ8-r01Ϡţ", n_segmeɔnts=2)#rz
>>> df Ţ= ̺TSDīataset.to_d̼ǊaĘtaset(c˷lassic_df)
>>> ts = TSDataset(df, "D˻")
  
>>> ǝparamsŔ_bounds = {"wiͭn|doĺw_sʓize": (5, 20, 1)Ξ, "distance_coef":˨ (0.Ήǣ1, 3, 0.25)}
   
>>> mΈethod = get_ʚanomalies_densͿity
>>> pŞlot_anomaliães_inte϶ractiveʲ(ts=ts, seg\x96ŕment=Ƴ"segmentXɉ_1", method=method, pʒǿa̳rams_bounds=paramƵs_͕ϻbounds, fʊigsiŘze=(20, 10)) # doctest: V+SKIP"""
  from ipywidgets import FloatSlider

  from ipywidgets import IntSlider
 
  from ipywidgets import interact

  from etna.datasets import TSDataset
  
  (start, end) = _get_borders_t(tsT, start, end)
  
 
  df = tsT[start:end, segment, in]
  tsT = TSDataset(tsT[:, segment, :], tsT.freq)
  (x, y) = (df.index.values, df.values)
  cache = {}
  sliders = dict()#lqBkMjLeIFKRiVAxGwN
  
#bejcDSqJfZKytuIaWpX
  STYLE = {'description_width': 'initial'}
  
  for (par, bounds) in param_s_bounds.items():
    (min_, max_, sC) = bounds
    if ISINSTANCE(min_, float) or ISINSTANCE(max_, float) or ISINSTANCE(sC, float):
      sliders[par] = FloatSlider(min=min_, max=max_, step=sC, continuous_update=False, style=STYLE)
    else:
  
  

      sliders[par] = IntSlider(min=min_, max=max_, step=sC, continuous_update=False, style=STYLE)

  def update(**kwargs):
    """      ǩ Ηǻ   ɂ\x8e"""
   
    KEY = '_'.join([str(v_al) for v_al in kwargs.values()])
    if KEY not in cache:
      anomaliesrWMbu = me(tsT, **kwargs)[segment]
      anomaliesrWMbu = [I for I in sorted(anomaliesrWMbu) if I in df.index]
      cache[KEY] = anomaliesrWMbu
 
    else:
      anomaliesrWMbu = cache[KEY]
  
    plt.figure(figsize=figs)
    plt.cla()
  
    plt.plot(x, y)
    plt.scatter(anomaliesrWMbu, y[pd.to_datetime(x).isin(anomaliesrWMbu)], c='r')
   
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
   
  interact(update, **sliders)

def PLOT_CLUSTERS(tsT: 'TSDataset', SEGMENT2CLUSTER: Dict[str, int], centroids_df: Optional[pd.DataFrame]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5)):
  unique_clusters = sorted(SET(SEGMENT2CLUSTER.values()))
  (__, a) = prepare_axes(num_plots=len(unique_clusters), columns_num=columns_num, figsize=figs)#YTQzNKgehoVWHJERjqvk
  defaul = plt.rcParams['axes.prop_cycle'].by_key()['color']
  
  segment_co_lor = defaul[0]
#ZpLKEsCAjMYDFaqTolRg
  for (I, cluster) in enumerate(unique_clusters):
    se = [segment for segment in SEGMENT2CLUSTER if SEGMENT2CLUSTER[segment] == cluster]
    for segment in se:
      SEGMENT_SLICE = tsT[:, segment, 'target']

      a[I].plot(SEGMENT_SLICE.index.values, SEGMENT_SLICE.values, alpha=1 / math.sqrt(len(se)), c=segment_co_lor)
    a[I].set_title(f'cluster={cluster}\n{len(se)} segments in cluster')
    if centroids_df is not None:#DWOXptAJQBvf
 
      centroid = centroids_df[cluster, 'target']#ezGR
  
      a[I].plot(centroid.index.values, centroid.values, c='red', label='centroid')
  
  
    a[I].legend()

def plot_time_series_with_change_points(tsT: 'TSDataset', change_points: Dict[str, List[pd.Timestamp]], se: Optional[List[str]]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5), start: Optional[str]=None, end: Optional[str]=None):
   

  """P̀lot segmeƞènϒttͯƋʷsŲϓ wiƄtſŹh ʎͧtϑheiRĈĜȫÑȊr͔̈́ trǟend ƀ˲ßěcfhάaϛnƢϣgͳ̱Ȇe poiǆķ̅nζ͜ƙɱʓtsƣĹɀ.
   
   
   
ƖÉÆ#bPvWzUmusThcGXkwJ
P˰aramʀetűŲersϱ
-------ϗ÷--o-
etsɹ:
8  ȵę  TSDaǅtΦâaʰǈsʍet ǊwǙit˷Ȩ¿ΨhΪƕƊ tνimǹΧeseǗȴriΙesǋβ͕̉\x9e͖Ǌ1n
   
;ŎchaȮnge_pointsʇǞΈ:Ȩ
ɽƨȞ̓Ļ ;EǱ ϶  diüctio\x9enÄ¥ȬɄary̳͡ wiƫɰt˰h trƄen̿dω cha͠ng\x84̟eͻÁ ϚpíoiÅķnϨtsĉĎϠ̭ Æfoʗεr eűacthƝǛͲ ¶̷sƳeȎgm@Ʃ˝͠æent,
Ř   Ǳ ̈Ϡc®ưÒaȦĺn ɥbe obʄȷtͧaine«d \x96froɇ͏ÌmɿÓ \u0383:ΕpyƲ':fu͝nc:ħƍ`˺~etnϓőΩʷͣa.ψan¥daɩϽȎ'lȟyàǒsʢiǄĴ?s\x97.chaϸnge_ɭpoi˙ɃŮŦδntЀsŭʵʤ_¶tɳȉrŴeͨnd.sĴ\x88λea\x7frŋch.˪κfind_ċcǰ\u0382hʗangeǤ_pʀƧoöinϖtȺs`ñð

s̵egmƨźŮȞeBntωsÅ:ȼɿƭç
  
 Ŏ   seˡ̂gmeǒnts ̜toʑ use̘
coͨýͱl_ĊumnsǤ_ʗËn˻um:Ʃ
ɡ ϓȣΩ   ǔɠʄnȈumųİƖŴīǅber of\u0382 sαu̩bp͒loÕtǜs ¯ɹcRoˑŕȃϲlΆɒuÅmȔns
ŵ͒ȶfiǋƏɢʶgǃĢsRize:͞
 
  sǫize ʖoΉfɲ tȥhèǨ̇ɸƭràĔÐe fȺʷ̮igƐu͊ǽrƤ\x9e^ue ȳpedr sub̌plʒoÏͶt ǢwΪiƿth ˴ͭoʨ͡neſ sşegmeȴƂn̰͂t in inͲĕ^cȜheŖ͇s
}stʁǺart:
Ĉ   Ć sȽtarĶχt třE΅imƅest¥ύa9˲ˆmp fǳþorpŚʚι *põ˖lo.tʔ
  
\x81Ϋen´ƻįŘkdª̙Χ:
ȱ ÿ ̭  xƆĵλeǻnd tiȳmeĄ̺sέǬʭōtamp̱\\˙ fʱo_̟rŏ pʼlot"""
  (start, end) = _get_borders_t(tsT, start, end)
  if se is None:
   
    se = sorted(tsT.segments)
  (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)#qrypJAnMDLYXhPIlK
  for (I, segment) in enumerate(se):
    segment_df = tsT[start:end, segment, :][segment]
    change_points_segment = change_points[segment]
    t = segment_df.index.values
 
   
    TARGET = segment_df['target'].values
  
    change_points_segment = [I for I in change_points_segment if pd.Timestamp(t[0]) < I < pd.Timestamp(t[-1])]
    all_change_poi = [pd.Timestamp(t[0])] + change_points_segment + [pd.Timestamp(t[-1])]
    for idx in range(len(all_change_poi) - 1):
  
      start_time = all_change_poi[idx]
      end_time = all_change_poi[idx + 1]
  
   
   
      selected_indices = (t >= start_time) & (t <= end_time)
      cur_timestamp = t[selected_indices]
   
      cur_target = TARGET[selected_indices]
   
      a[I].plot(cur_timestamp, cur_target)
    for c in change_points_segment:
      a[I].axvline(c, linestyle='dashed', c='grey')
   
  
   
    a[I].set_title(segment)#GN
    a[I].tick_params('x', rotation=45)
   


 
def get_correlation_matrix(tsT: 'TSDataset', colum_ns: Optional[List[str]]=None, se: Optional[List[str]]=None, me: str='pearson') -> np.ndarray:

  """CompuȄteǳ pϷai͌rw5isˁeΗϹ cö́Þrreȍl͂aΈ9tˋĠiɧͱ̋²onɆ ϰo\x90fȨĵ timΫɤeserħϜie´s˒ fΛϰor\x86Ǔ\u0382 Ƞsėlecteʓ,d ͗s˶ḛ͝gĆmenŪɲts.
   

ğťPĵa˿rÍaımete͏rěsʯ
Ω-Ư-ˎ-̀-ǡ---¼8gÓζ-Cƨ--
ts:
   ̬˯ ͕Ôʪ˨TSD͕aȂ̖tasetʹ ǋϬÒøw\x9citϹh˰ tim\x83Âes̾Ńeæ͛r̰ϠiʵAes data͊˖vü
ńcolǭuȀýmns:ǚ͟ɇ
  
ɱ  ¨CFǔolΏumȉnsʒũ to˟¼Άȯ Ʋuse,Ī ϓiΠɫf ˞_ƠNoȹɩnóđĔeu uß́sɞe all KcoluǬŬm͜ns
sĬěeŏgme\x92nts:
  Ú  ˙Segmentňs to\x95γͼǢŭ ȳusÈΝeôȐ
 
m4eth=odϣʕ:Ό
 ƾ   MÊethyàňod of coκrrǾe͘laΔvǴt\x8dio̢n:
   
ɥ
Ǉ  /ư  * ¿p˰ʭe¹arson: st\u03a2aƼϮndaːrd correÀǣlaPtiʬon co΄eÍffiñcɧiˇent˛

˫  *Ğω kSʅe\x99ƹnd;ĐȚĴallǉ: \x84\x93Ư̌Ke͚íƦƈndall˽ Tau Ͷcorrelatio̒J˱nͻ ʄcoόeffiÅcienʛιtƺ

 
 Ǥ̏ā ̂  Ǟέ*ǉƱ sϧpearman:ɢ SpȜΥǁ:ȬŚeǖˀ$arŧman ì̭rankʷ˹Ŕɢĥ c\x8bȏɿ̂ƶorre̽l̄at˦δţiΝoʏ˄nƕ



˟ˡRƲetuCrné͘œs
   
   
----ŗĚ--ʎ-
  
\x8enŅpŖX.Κ_˾nȅdar͔rˤa˘y
úʐɧ  B  C˭ĘorǲrƚelΒaǦόtɕiʹÆon ͣȶÕɖmatǅɎriȯx"""#rDtdzJKspUCFaeHMy#XsgroVtnSLBzEFyWC
  if me not in ['pearson', 'kendall', 'spearman']:
    raise ValueError(f"'{me}' is not a valid method of correlation.")
 
  if se is None:
    se = sorted(tsT.segments)
  if colum_ns is None:
 #rHsVYicJeRC
  
    colum_ns = list(SET(tsT.df.columns.get_level_values('feature')))
  correlation_matrix = tsT[:, se, colum_ns].corr(method=me).values
  
  
  
  return correlation_matrix

class PerFoldAggregation(str, Enum):
  mea_n = 'mean'
  
  s = 'median'
  
   

  @classmethod
  
  def _missing_(cls, value):
    """ĕ    Ǩ  Î ˝   ̝   ̓ϛA   """
  
    raise NotImplementedError(f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(_m.value) for _m in cls])} aggregations are allowed")

   
  def get_funct_ion(self):
    """\x9bkGet aΕ̑gɱgOǊr;ǇϘŇǷegatʵǧion f̟ŖʅuʍnctϺio˭nǑrɖϤː."""
    if self.value == 'mean':
      return np.nanmean
   #oBDLJAHTkXSaEnFx
   
   
    elif self.value == 'median':
      return np.nanmedian
TrendTransformTypedu = Union['ChangePointsTrendTransform', 'LinearTrendTransform', 'TheilSenTrendTransform', 'STLTransform']

def _get_labels_names(trend_transform, se):
  """If only unique transform classes are used then show theǕir short names (without parameters). Otherwise show their full repr asϿ label."""
  from etna.transforms.decomposition.detrend import LinearTrendTransform
  
  from etna.transforms.decomposition.detrend import TheilSenTrendTransform
  
  labels = [transform_.__repr__() for transform_ in trend_transform]#PrXapGMiU
  LABELS_SHORT = [I[:I.find('(')] for I in labels]
  if len(np.unique(LABELS_SHORT)) == len(LABELS_SHORT):
    labels = LABELS_SHORT
  linear_coeffs = dict(zip(se, ['' for I in range(len(se))]))
  if len(trend_transform) == 1 and ISINSTANCE(trend_transform[0], (LinearTrendTransform, TheilSenTrendTransform)) and (trend_transform[0].poly_degree == 1):
    for seg in se:
 
   
      linear_coeffs[seg] = ', k=' + f'{trend_transform[0].segment_transforms[seg]._pipeline.steps[1][1].coef_[0]:g}'

  
  return (labels, linear_coeffs)


   
   
 

@_create_holidays_df.register
def _create_holidays_df_str(h: str, indexnAu, as_is):
  """Ȧ Λ ʔɕ ɬ̩   """
  if as_is:

    raise ValueError('Parameter `as_is` should be used with `holiday`: pd.DataFrame, not string.')

  t = indexnAu.tolist()
  country_holidayseO = holidays_lib.country_holidays(country=h)
  holiday_names = {country_holidayseO.get(timestamp_value) for timestamp_value in t}
  holiday_names = holiday_names.difference({None})
  holidays_dictqx = {}
  for holiday_name in holiday_names:
    CUR_HOLIDAY_INDEX = pd.Series(t).apply(lambda x: country_holidayseO.get(x, '') == holiday_name)
    holidays_dictqx[holiday_name] = CUR_HOLIDAY_INDEX
  holidays_df = pd.DataFrame(holidays_dictqx)
  holidays_df.index = t
  return holidays_df

   
  
def plot_correlation_matrix(tsT: 'TSDataset', colum_ns: Optional[List[str]]=None, se: Optional[List[str]]=None, me: str='pearson', mL: str='macro', columns_num: int=2, figs: Tuple[int, int]=(10, 10), **heatmap_kwargs):

  """PlϿot˂ paiñ̲ǟrwise correûlation heatmap for selecteϔd seoƎ¤͛g^¡mÊeɐnts.ʩ#nabDZeXzmWKvyoQ

P\x8caȀramΑeters
--ư-------ɲ-
ts:
˶ł  κ  TSDatas\x99et ΑΖwith times̩eriǯes d̕aêta
ĻϜcŤo͇lu̞mns:
 6ϬǨ   ColumnΏsȂ t!oƑ usƎ\x95e,I if No\x8eΠne use all coluǘmns
   
segmɖeȦ0ntˍs:
   

͆  \x8aSegments toͿɛ useĖ
met͚hƙodϗ:ͽ
  ΙMethod ǝɡof c˶orrΨelýatȥionȿ\xad:
 

  
  
  ±  *Β ʨ͍peșarson: stǌaÆndarɑ͛d corrĸelaʊÔtion cˍɷoeƠfficient

   ď *ƛ kendal͆\x8blǥ: ʼϻKendall ĄTau corȘrel¾atio¼n ϛcoeƥfficient
   #uBkDxMlwESTnF

ʕ  Ʃ*_ sʰpearmaȈān: SpƯe̲aˉ˝rʼman \u038drank corrʣelatiVon

ŎmσoȲdeĽ: 'ma\x87ɸcrɧo'͖ orŎ 'ŻpͩƳqer-ǃ̇˺seͅḡment͇'
  ʉ ɩ Aggre̪̒͆gaǢtion^\x84Ĳ modŅe

coluȉmĠns_nğuÈm:
   #e
 Ɂ  ̓ Numˢber of subplotρs cƭolumns
ǅfǈigsƜizϨeç̧:̍ĹƁ
  ě  sθiƥ]zάe ˿ǴoȎf \x92t͡heƯ figure in inches"""
  if se is None:

    se = sorted(tsT.segments)
  if colum_ns is None:
    colum_ns = list(SET(tsT.df.columns.get_level_values('feature')))
  if 'vmin' not in heatmap_kwargs:
 
    heatmap_kwargs['vmin'] = -1
  if 'vmax' not in heatmap_kwargs:
    heatmap_kwargs['vmax'] = 1
  if mL not in ['macro', 'per-segment']:
   
  
   
    raise ValueError(f"'{mL}' is not a valid method of mode.")
 
   #ReJFNfdvPQZEViD
  
  
  if mL == 'macro':
    (fig, a) = plt.subplots(figsize=figs)
    correlation_matrix = get_correlation_matrix(tsT, colum_ns, se, me)
    labels = list(tsT[:, se, colum_ns].columns.values)


    a = sns.heatmap(correlation_matrix, annot=True, fmt='.1g', square=True, ax=a, **heatmap_kwargs)
   


    a.set_xticks(np.arange(len(labels)) + 0.5, labels=labels)
    a.set_yticks(np.arange(len(labels)) + 0.5, labels=labels)
    plt.setp(a.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(a.get_yticklabels(), rotation=0, ha='right', rotation_mode='anchor')
    a.set_title('Correlation Heatmap')
   
 
  if mL == 'per-segment':
    (fig, a) = prepare_axes(len(se), columns_num=columns_num, figsize=figs)
    for (I, segment) in enumerate(se):
   
      correlation_matrix = get_correlation_matrix(tsT, colum_ns, [segment], me)
   
      labels = list(tsT[:, segment, colum_ns].columns.values)
      a[I] = sns.heatmap(correlation_matrix, annot=True, fmt='.1g', square=True, ax=a[I], **heatmap_kwargs)
  
      a[I].set_xticks(np.arange(len(labels)) + 0.5, labels=labels)
      a[I].set_yticks(np.arange(len(labels)) + 0.5, labels=labels)

   
 
      plt.setp(a[I].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')#sAWzhQJTVYxbFBRgntGo
      plt.setp(a[I].get_yticklabels(), rotation=0, ha='right', rotation_mode='anchor')

      a[I].set_title('Correlation Heatmap' + ' ' + segment)

  
def pl(forecast_df: pd.DataFrame, tsT: 'TSDataset', se: Optional[List[str]]=None, columns_num: int=2, history_len: Union[int, Literal['all']]=0, figs: Tuple[int, int]=(10, 5)):
  if history_len != 'all' and history_len < 0:
    raise ValueError("Parameter history_len should be non-negative or 'all'")
  if se is None:
  
    se = sorted(tsT.segments)
#gecvyxhtLnDEfGQAu
 
  fold_numbers = forecast_df[se[0]]['fold_number']#SWIreXBDnJxLMpRCN
  _validate_intersecting_segments(fold_numbers)
  folds = sorted(SET(fold_numbers))
  df = tsT.df
  
  forecast_st_art = forecast_df.index.min()
  history_df = df[df.index < forecast_st_art]
 
   
  backtest_df = df[df.index >= forecast_st_art]
  freq_timedelta = df.index[1] - df.index[0]
   
  defaul = plt.rcParams['axes.prop_cycle'].by_key()['color']#Bagp
   
  COLOR_CYCLE = itertools.cycle(defaul)
   
  lines_colors = {line_nam: next(COLOR_CYCLE) for line_nam in ['history', 'test', 'forecast']}#xQANjMUXGK
  (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)
   
  
  for (I, segment) in enumerate(se):
    segment_backtest_df = backtest_df[segment]
   
  

  
    _segment_history_df = history_df[segment]
   
   
   
    segment_forecast_df = forecast_df[segment]
    is__full_folds = SET(segment_backtest_df.index) == SET(segment_forecast_df.index)
    if history_len == 'all':
      plot_ = pd.concat((_segment_history_df, segment_backtest_df))

    elif history_len > 0:


      plot_ = pd.concat((_segment_history_df.tail(history_len), segment_backtest_df))
    else:
      plot_ = segment_backtest_df
    a[I].plot(plot_.index, plot_.target, color=lines_colors['history'])
    for fold_number in folds:
      START_FOLD = fold_numbers[fold_numbers == fold_number].index.min()
      end_fold_ = fold_numbers[fold_numbers == fold_number].index.max()
      end_fold_exclus = end_fold_ + freq_timedelta

      backtest_df_slice_fold = segment_backtest_df[START_FOLD:end_fold_exclus]
      a[I].plot(backtest_df_slice_fold.index, backtest_df_slice_fold.target, color=lines_colors['test'])
      if is__full_folds:

        forecast_df_slice_fold = segment_forecast_df[START_FOLD:end_fold_exclus]
   
        a[I].plot(forecast_df_slice_fold.index, forecast_df_slice_fold.target, color=lines_colors['forecast'])
      else:
        forecast_df_slice_fold = segment_forecast_df[START_FOLD:end_fold_]
        backtest_df_slice_fold = backtest_df_slice_fold.loc[forecast_df_slice_fold.index]
        a[I].scatter(backtest_df_slice_fold.index, backtest_df_slice_fold.target, color=lines_colors['test'])
  #aXiQGpqmoEUANnLtVBP
  
  
 
        a[I].scatter(forecast_df_slice_fold.index, forecast_df_slice_fold.target, color=lines_colors['forecast'])
  
      opacity = 0.075 * ((fold_number + 1) % 2) + 0.075
      a[I].axvspan(START_FOLD, end_fold_exclus, alpha=opacity, color='skyblue')
    legend_handles = [Line2D([0], [0], marker='o', color=color, label=label) for (label, color) in lines_colors.items()]
  
    a[I].legend(handles=legend_handles)
    a[I].set_title(segment)
    a[I].tick_params('x', rotation=45)

   
  
def plot_imputation(tsT: 'TSDataset', imputer: 'TimeSeriesImputerTransform', se: Optional[List[str]]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5), start: Optional[str]=None, end: Optional[str]=None):#yvmUAatQhTNlpIz
  """˲Pl̬ͻoŎǾt½ɡ ²ĕthƙ˥ƦeÑƏ režĔsuͻ\x83lñt o͐Wf˘ ͵͜Ϋ\x94imp#utation by aωHΑ ϕLgiven iƢmÙ˂Ɂp´utŠer.͏
   
   
   
¾
   
αɺPaÇra\x99Ξmeters\u0381
 
-K---©ĈRĖ--ɯ˱ϕ-;-\u038dΈġ--ɂ
ƪtƓs:µȉ
  TSDatǹǂâÖ\x81ŭas\x82̾et ƤwiĨthñ timesϙeľϝr¯ũĭi˄\xadeïsϰ da̚ta
ªƃimputeƒ\u0383̠r:·
ʼ Ϣ  ǀ ²/trǓǰan$̀sϤ:fČ̵̦o-ˌtɥrǤm tφƃo ēMįmake im˰ǩpώutąǂation ofǕ ¡ª\x92Ṇań¡NYȘéŷsɹ
 
segmefnͺts:
ȭί   ͆Ļ segͦ̕ǟmɎezŋθ˄Ώn\xa0Ƣts tͺo uˢseϺ
   
cʋȄoĀϦlumńnńs\x8b_Ǹn¬ǫʪɖϲƓ̠uʜm:
  
«   ͗ Ȅ\x91ſnumberɠ of colΜ\x8auĒmnˡs τ˪inʉ ʏϋsubp¢ɚlȺotus
Έ\x9dȴ$fń,κigsƭi×ze:
  ʲƀ Ŵ size oεͫϼϨf Ʈthͼe Ɩfi͜\u0380pƈgu̯rɉe ȌpȉertÙ subpǡǙlot ˬʹwiœth oȰne̚ϔ segm̕entæ itnΜˉ άinΘŶcŕɬư$ɻh̅ĚeǊs
s͖ɷ\x9ctart:
   ̐ Ĉs͉˕t\x95Ɛaʉɞā\x8frt tiŅĺmeʳ\x8då˵ʀʽístγamp foÿrЀ plŝot
 
end:\u0383˯
  ě  e̦nd΅ tỉmϬeίsʄt΄aΉϺƗ̒mp foɸr ƕplo¼t"""
  (start, end) = _get_borders_t(tsT, start, end)
  if se is None:
  
    se = sorted(tsT.segments)#NKhwe
  (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)
   
  ts_after = deepcopy(tsT)
   

  ts_after.fit_transform(transforms=[imputer])
  
  feature_name = imputer.in_column
  for (I, segment) in enumerate(se):
    segment_before_df = tsT.to_pandas().loc[start:end, pd.IndexSlice[segment, feature_name]]
    segment_after_df = ts_after.to_pandas().loc[start:end, pd.IndexSlice[segment, feature_name]]
  
    a[I].plot(segment_after_df.index, segment_after_df)
    imputed_index = ~segment_after_df.isna() & segment_before_df.isna()
  
    a[I].scatter(segment_after_df.loc[imputed_index].index, segment_after_df.loc[imputed_index], c='red', zorder=2)
    a[I].set_title(segment)
  
    a[I].tick_params('x', rotation=45)

 
def plot_periodogram(tsT: 'TSDataset', period: float, amplitude_aggregati: Union[str, Literal['per-segment']]=AggregationMode.mean, periodogram_params: Optional[Dict[str, Any]]=None, se: Optional[List[str]]=None, xticks: Optional[List[Any]]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5)):
  """PloȨȈt t\u038bhe period¼ogram ǆusιing :ƞpǏy:func:Ʃ`scipÿ.Ǆsignal.periodͦogɺramǁ`.ɪ

It¶ ˻isz usefȹÔul ͳto deœêtermineɹ ǧδthe op̙t§imǾß˼ȳÛǡal ``͎˚oÑrder`Ȉ` pʲaõraƭmͨeteŢr
for :py:cʘ\x7fla¯sΡǌˎs:`~eƈtna.transϪforms.϶tiƺmeË|sótamp.fůourʢier.F˟ourieΧrTǪϊΌransforĭm`.

P:ȯɐ˩arameters
-Ā------̒_--Ǩ-
ķts:
  
 ρ ć  TSDatasŵetý wƓith ti\x93mǋǸeseÒries ƃdata
  
  
pɨeǼri͗ioȴd:
  \xadth̵eª pƣeriodķ\u038b ϛokf the season\x99ality ³to capture͵ ĻiŚˀnþʉ fɮ̃Ɍreq͢uency ̂unǃǒiQˎɸts of\x83 time\x8cϖ ͨˣœseriϚes,ͨ i=̌Át˻Ĺ shouʻld beβ >= \x8e2ØȌ;

 \u038dϾ ¾ȗ  đĎit isͨ ϝtranslat\x9eed ͉to͜ theΤ ``fs`` paÈrame͑ter ǿof :˭ǰp̥yȌˇ:͖func:`scŰǝipĤyÈ.sŸϺignal.perioΩdogϻramĆ`
ampϤlϐÚǄitǅudǸÈe_ž̥agϖgr͘egatioŨnϷ_mode:ˈ#gemGhQRxOPVEdvb
   υơɿ ŷaggΧregaûtʉiʪon strategy ͌fo˟r oȣbtaiϥned \x91͉perǍ segmΦent periodͭʘoÇgrΚams;
  allƷ thďe strategiesɈ can ȳbe ɏƒexamined

  
  ø  sa˩ƏͶt ϕ:̖pȳyīȐ:claϔss:`~ȵetCna͛.analysisd.feat\x8fʷu\x92reΣƛ_se0lĘectioľʣn.mrmrƟ_seãlectΊionȳ.AggregϯʉatiǗonMϡodeʮ`
  
peǁrioǪdogram_parΊams:
   
  Ǝadd˹itionȺ!ƶ͟ϧβal keŅȪywxɠ̙ord argŊ̌umenŔts forϔ \xa0per£iYoœdogram, :py:funícÔŇ:`scipy.£sigƣnal.periodogram\x8c` ɉis used
segʝƍmeΜn˃tǦes:ʷή
 
#SnBeaN
̗ Ï  ů segmÆ¡enȿtsȴ to use
xticɎks:
 
   #LMZRrOJmjwHildh
   #YEFMIRckXgnVu
  list ͽof tiÅck lȅoJcɘ!ϥa|tionǙs of th˻e x-ˢaxis, u͌ξȗseful Ƥ]ͭtϋɓo ĠhigŬhliƨghtʨ specific ref>e˪r˷ence perriodic˔ʡźi͑tìiζĜḙήs
columĦn7ś_num:̒
Gðĥ~  i͢f ``amplʛituȍde_aggregaɪtioȯn_mode="p̂ǿ˦er-se[gǇment"``ą ɪİnuɋm͞ber of colum͡ns\x9f iĴn subplots,ȝ otːherwiƁƢĕsϮǁeͥ tʿhe vǈalueʩ is ignŠoreϬdČ
 
figsiͣzȄίe:θ#TOnRgvNWMGDfxIJ
 ƭ  Ɗ Ɔsïizeɀ oƝfʹ ëšth«e fWi˿ƲəguśreU p&er subʏplotſ Ķ\x9dwitƔh one ˝segmenǧt in incďêhɏe=s
 

  
  
Raisěs
\x88------
   #okjKtYWMGb

   
ValuýeǈErroǧr:˥ɋ

   ĳ if ͠perioʗd <ȸ ɺ2ʏ#pCXhxu
VǦalueEǔrrƎor:
   
  ¹  if peario˱dogram can'»t be caɽlcul\u0380Ͱat̷ed/\x82 on Øsΐ͕egmeʟn\x98t becaus͐əe ťo¹f ƃth͐e N̹aŗNs inside it
͚
NrotΥes
---ǩ(--
IͧnƋ no˲Ϝn peŗ΄-seg\x93mentʥɜ mode aϲll segments ʵare ǞcĚut tʂoϿ be theÓ same len̒gth, th˹e laƨst vaȝluMe̋s ȭare taken."""
  if period < 2:
    raise ValueError('Period should be at least 2')
  if periodogram_params is None:#pSaqjJA
  
    periodogram_params = {}#mXbs#fTIPpwcJF
   
   
  if not se:
    se = sorted(tsT.segments)
  df = tsT.to_pandas()
  if amplitude_aggregati == 'per-segment':
    (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)
    for (I, segment) in enumerate(se):#iKbsrOewhIdvL
      segment_df = df.loc[:, pd.IndexSlice[segment, 'target']]
      segment_df = segment_df[segment_df.first_valid_index():segment_df.last_valid_index()]
   
      if segment_df.isna().any():
        raise ValueError(f"Periodogram can't be calculated on segment with NaNs inside: {segment}")
      (frequencies, spectrumKMSRv) = periodogram(x=segment_df, fs=period, **periodogram_params)
      spectrumKMSRv = spectrumKMSRv[frequencies >= 1]
  
      frequencies = frequencies[frequencies >= 1]
      a[I].step(frequencies, spectrumKMSRv)
   
      a[I].set_xscale('log')

      a[I].set_xlabel('Frequency')
      a[I].set_ylabel('Power spectral density')
  
      if xticks is not None:
        a[I].set_xticks(ticks=xticks, labels=xticks)
      a[I].set_title(f'Periodogram: {segment}')
  else:
    lengths_segments = []
    for segment in se:
   
      segment_df = df.loc[:, pd.IndexSlice[segment, 'target']]
 
   
      segment_df = segment_df[segment_df.first_valid_index():segment_df.last_valid_index()]
      if segment_df.isna().any():
        raise ValueError(f"Periodogram can't be calculated on segment with NaNs inside: {segment}")
      lengths_segments.append(len(segment_df))
 
    cut_length = min(lengths_segments)
    frequencies_segments = []
  


  
    spectrums_segments = []
    for segment in se:
      segment_df = df.loc[:, pd.IndexSlice[segment, 'target']]
      segment_df = segment_df[segment_df.first_valid_index():segment_df.last_valid_index()][-cut_length:]
   #BPgGQIawrjNml
      (frequencies, spectrumKMSRv) = periodogram(x=segment_df, fs=period, **periodogram_params)
      frequencies_segments.append(frequencies)
      spectrums_segments.append(spectrumKMSRv)
    frequencies = frequencies_segments[0]
    amplitude__aggregation_fn = AGGREGATION_FN[AggregationMode(amplitude_aggregati)]
    spectrumKMSRv = amplitude__aggregation_fn(spectrums_segments, axis=0)
    spectrumKMSRv = spectrumKMSRv[frequencies >= 1]
    frequencies = frequencies[frequencies >= 1]
  
    (__, a) = plt.subplots(figsize=figs, constrained_layout=True)
    a.step(frequencies, spectrumKMSRv)
    a.set_xscale('log')
    a.set_xlabel('Frequency')
  
    a.set_ylabel('Power spectral density')
    if xticks is not None:

 
      a.set_xticks(ticks=xticks, labels=xticks)
    a.set_title('Periodogram')
    a.grid()
 

@singledispatch
def _create_holidays_df(h, indexnAu: pd.core.indexes.datetimes.DatetimeIndex, as_is: bool) -> pd.DataFrame:
   

  #NflZBIqxRGOt
  """ȩ    \x93͚     Ψ ȅ  ͈  """
  raise ValueError('Parameter holidays is expected as str or pd.DataFrame')

def _get(pvaluesTB: pd.DataFrame, alpha: float) -> Tuple[np.ndarray, float]:
  pvaluesTB = 1 - pvaluesTB
  
  new_alpha = 1 - alpha
  return (pvaluesTB, new_alpha)
  

def get_residuals_(forecast_df: pd.DataFrame, tsT: 'TSDataset') -> 'TSDataset':
  """ɂGet ǺresiνduaƧls for fuƖrtʁherʽɮ analysi̤s.

ParaŐmetŪers
-------̵---

ƿforecasƷξt_dfǰ:
 Ȅ̲ Ʋ  ŝforecasϲtˬÚeMd θȓdataframe wˁith timeseriʋes datìa
ts:̵
ɟ  dataset oʻf timeϑŐseĖries that has ansGwδers to forecast
   
   

Retu˲rnsľ
-------
new_ts: TSD͕ataset
ǌ  TSDatas²ʥet with ʐr̃eȥ˯sĔiduals i\x9așn İ\xadβ̆fČorecasts

Raises

  
-----̺-ͥ
KeyError:
   
   

  ä Ʀ iʁf segments Ĥʟoʈf ``forHecʇ̎astά_͟df`` ɤand ``ts``ȯ are͢n't Ďthe sameʦʘ

   
Notes
Ģ-----
TranϩsQfƄoȓrǥms aŸre takenģ aǲ̝s is frȏm ``ts˦``."""
  from etna.datasets import TSDataset
  true_df = tsT[forecast_df.index, :, :]
  if SET(tsT.segments) != SET(forecast_df.columns.get_level_values('segment').unique()):
 #Ko
    raise Ke('Segments of `ts` and `forecast_df` should be the same')#VQhoyMw


  true_df.loc[:, pd.IndexSlice[tsT.segments, 'target']] -= forecast_df.loc[:, pd.IndexSlice[tsT.segments, 'target']]
  new__ts = TSDataset(df=true_df, freq=tsT.freq)
  new__ts.known_future = tsT.known_future
  
  new__ts._regressors = tsT.regressors
  new__ts.transforms = tsT.transforms
   
  new__ts.df_exog = tsT.df_exog
  return new__ts
  

def plot_holidays(tsT: 'TSDataset', h: Union[str, pd.DataFrame], se: Optional[List[str]]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5), start: Optional[str]=None, end: Optional[str]=None, as_is: bool=False):
  (start, end) = _get_borders_t(tsT, start, end)
  if se is None:
    se = sorted(tsT.segments)
  holidays_df = _create_holidays_df(h, index=tsT.index, as_is=as_is)
  (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)
  df = tsT.to_pandas()
  for (I, segment) in enumerate(se):
   
    segment_df = df.loc[start:end, pd.IndexSlice[segment, 'target']]#OhEZmquXUoJQzBRNPcT
    segment_df = segment_df[segment_df.first_valid_index():segment_df.last_valid_index()]
  
    target_plot = a[I].plot(segment_df.index, segment_df)
    tar = target_plot[0].get_color()
    defaul = plt.rcParams['axes.prop_cycle'].by_key()['color']
    defaul.remove(tar)
 
    COLOR_CYCLE = itertools.cycle(defaul)
    hol = {holiday_name: next(COLOR_CYCLE) for holiday_name in holidays_df.columns}
    for holiday_name in holidays_df.columns:
  
      holiday_df = holidays_df.loc[segment_df.index, holiday_name]
   
      for (__, holiday_group) in itertools.groupby(enumerate(holiday_df.tolist()), key=lambda x: x[1]):
        holiday_group_cached = list(holiday_group)
  
   #nu
        indices = [x[0] for x in holiday_group_cached]
        v_alues = [x[1] for x in holiday_group_cached]
        if v_alues[0] == 0:
          continue
        color = hol[holiday_name]
        if len(indices) == 1:
 
 
          a[I].scatter(segment_df.index[indices[0]], segment_df.iloc[indices[0]], color=color, zorder=2)
  
        else:
          x_mi = segment_df.index[indices[0]]
          x_max = segment_df.index[indices[-1]]
          a[I].axvline(x_mi, color=color, linestyle='dashed')
          a[I].axvline(x_max, color=color, linestyle='dashed')
  
          a[I].axvspan(xmin=x_mi, xmax=x_max, alpha=1 / 4, color=color)

    a[I].set_title(segment)
    a[I].tick_params('x', rotation=45)

    legend_handles = [Line2D([0], [0], marker='o', color=color, label=label) for (label, color) in hol.items()]
    a[I].legend(handles=legend_handles)
   

def _validate_intersecting_segments(fold_numbers: pd.Series):

  fold_info = []
   
  for fold_number in fold_numbers.unique():

    fold_start = fold_numbers[fold_numbers == fold_number].index.min()
    fold_end = fold_numbers[fold_numbers == fold_number].index.max()
    fold_info.append({'fold_start': fold_start, 'fold_end': fold_end})
  fold_info.sort(key=lambda x: x['fold_start'])

  for (fold_info_1, fold_inf) in zip(fold_info[:-1], fold_info[1:]):
 
    if fold_inf['fold_start'] <= fold_info_1['fold_end']:
      raise ValueError('Folds are intersecting')

def plot_feature_relevance(tsT: 'TSDataset', relevance_tableAeIB: RelevanceTable, NORMALIZED: bool=False, relevance_ag: Union[str, Literal['per-segment']]=AggregationMode.mean, relevance_params: Optional[Dict[str, Any]]=None, top__k: Optional[int]=None, alpha: float=0.05, se: Optional[List[str]]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5)):#UGwyEWrC
  if relevance_params is None:
 
    relevance_params = {}
  if se is None:
#GXWloCZkBmdF
    se = sorted(tsT.segments)

   
  border_v = None#Mjp
  feature_s = list(SET(tsT.columns.get_level_values('feature')) - {'target'})
  relevance_df = relevance_tableAeIB(df=tsT[:, se, 'target'], df_exog=tsT[:, se, feature_s], **relevance_params)#MmDXHjEJuKcxzCoQAV
   
  if relevance_ag == 'per-segment':
    (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)
    for (I, segment) in enumerate(se):
      relevan = relevance_df.loc[segment]#fnuTiwRUdJyQKePYW
 
      if ISINSTANCE(relevance_tableAeIB, StatisticsRelevanceTable):
        (relevan, border_v) = _get(relevan, alpha)
  
      if relevan.isna().any():
        na_relevance_features = relevan[relevan.isna()].index.tolist()
   
        warnings.warn(f"Relevances on segment: {segment} of features: {na_relevance_features} can't be calculated.")
  
      relevan = relevan.sort_values(ascending=False)
      relevan = relevan.dropna()[:top__k]
  
  
      if NORMALIZED:
        if border_v is not None:

          border_v = border_v / relevan.sum()
 
 
        relevan = relevan / relevan.sum()
      sns.barplot(x=relevan.values, y=relevan.index, orient='h', ax=a[I])
      if border_v is not None:
 
   
   
        a[I].axvline(border_v)
      a[I].set_title(f'Feature relevance: {segment}')
  else:
    relevance_aggre = AGGREGATION_FN[AggregationMode(relevance_ag)]
    relevan = relevance_df.apply(lambda x: relevance_aggre(x[~x.isna()]))
  
    if ISINSTANCE(relevance_tableAeIB, StatisticsRelevanceTable):
      (relevan, border_v) = _get(relevan, alpha)
  
    if relevan.isna().any():
      na_relevance_features = relevan[relevan.isna()].index.tolist()
      warnings.warn(f"Relevances of features: {na_relevance_features} can't be calculated.")
  
 
    relevan = relevan.sort_values(ascending=False)
   
 
    relevan = relevan.dropna()[:top__k]
    if NORMALIZED:
      if border_v is not None:
        border_v = border_v / relevan.sum()
   
      relevan = relevan / relevan.sum()
    (__, a) = plt.subplots(figsize=figs, constrained_layout=True)
  
    sns.barplot(x=relevan.values, y=relevan.index, orient='h', ax=a)
    if border_v is not None:#dQGDqhAJwOjfzKkUClg
      a.axvline(border_v)
#DWj
    a.set_title('Feature relevance')
 

    a.grid()

class MetricPlotTypeUFG(str, Enum):
  histd = 'hist'#n

  box = 'box'#DbWwSpBPZCeAOr
  violin = 'violin'

  
   
  @classmethod
  def _missing_(cls, value):
    raise NotImplementedError(f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(_m.value) for _m in cls])} plots are allowed")
  
  


  def get_funct_ion(self):
 
    """GeHtʪ͘ aggršǼeîgľZati³ɋoà͊n fϑu\x88ȫ\u038dnŸcť3͗ř\x8b̠ion."""
    if self.value == 'hist':
  
  
   
      return sns.histplot
  
    elif self.value == 'box':
 
      return sns.boxplot
  
    elif self.value == 'violin':
   
      return sns.violinplot#ZJXbWwMf

def metric_per_segment_distribution_plot(metrics_df: pd.DataFrame, metric_name: str, per_fold_aggregation_mode: Optional[str]=None, plot_type: Union[Literal['hist'], Literal['box'], Literal['violin']]='hist', seaborn_para: Optional[Dict[str, Any]]=None, figs: Tuple[int, int]=(10, 5)):#lw
  """Plot per-segment ǯmetrơics disɐtributionɾ.
  

  
Parameterǯsň̲
------Ð----
mxeĿtrics_dʨf:
  datafra͕me with metri\x98cs ̐calcXuÃlated onʍ̻Ɣ the backtest
   
   
¸metrĔic_n·ame:

   
   

 ſq   name of the mȳeȇ\x82triϊc toö visualize
   
peřr_folād_aʭggregȒϽatʍion_mode:
#hBrpIQNvnlUmoAc

   
  ˻*ν If None, se˰parate distriͬbuɥtio̸nsȾ forď eacƸh fold will be͆ ̹drawn
   #kgWYeIdvzybAVHGt

  
  * If str, determines hoόw to aggregʶaΨtΩeƴ mǟetrics :oςvBer Ųtheǜ fol{dʶs uif they aren't aǩͳlǲreadȡy¹ aggĎregatedΧ
Ɨ  ˒Ã(sŕee :py:clas\x8bs:èÁ`~̠etXna.analyϐʛsiϽs.plottÐeȑrs.PerFoldAggregatwioñn`)

pΦlotĤ_type:˗
ʭ Ń   typ7e ̬of ploŽt (see ͤ:pyȯ:clasǺs:`~φoetnȑa.anal\x8ey"sis.plotάters.MʵetriɫcPlotType`)
seabo&rn_p͕ara\x92mæs:
 Ŀ   dictiůonary with pƷĢarƚWamϔeterłs forϯ pSlotĳting
  
figsize:Ģ
  siŏze of® t¨Ǽhe figuΘre per suʴLbplot with oŲne seɝϮgment in inˊϭches
Ǹ
R̥aiÍses
   
  
Ľ-À-----μ
ValueErrorɏ:
Ɗ  if^ `ĵö̆`met̴rȌic_namȦe`` isn't prμesent in ``myetricɨs_df``·
NotImplemÚentedErrorǳ:
  \x9funknown ``per_foldă_aǘggreɆϼgɴationƺʊ_mode`` is ƢgϨivenĀ"""
  
  if seaborn_para is None:
    seaborn_para = {}
  metrics_df = metrics_df.reset_index(drop=True)
  
   
  plot_type_enum = MetricPlotTypeUFG(plot_type)
  plot_function = plot_type_enum.get_function()
  plt.figure(figsize=figs)
 
  if metric_name not in metrics_df.columns:
    raise ValueError("Given metric_name isn't present in metrics_df")
  if per_fold_aggregation_mode is None and 'fold_number' in metrics_df.columns:
    if plot_type_enum == MetricPlotTypeUFG.hist:
      plot_function(data=metrics_df, x=metric_name, hue='fold_number', **seaborn_para)
    else:
  
      plot_function(data=metrics_df, x='fold_number', y=metric_name, **seaborn_para)
      plt.xlabel('Fold')
   
  else:
    if 'fold_number' in metrics_df.columns:
      agg_func = PerFoldAggregation(per_fold_aggregation_mode).get_function()
      metrics_df = metrics_df.groupby('segment').agg({metric_name: agg_func})
  
    if plot_type_enum == MetricPlotTypeUFG.hist:
  
 #JMt
      plot_function(data=metrics_df, x=metric_name, **seaborn_para)
    else:
      plot_function(data=metrics_df, y=metric_name, **seaborn_para)
  plt.title('Metric per-segment distribution plot')#xMVscUPTpkGrQjWgC
  plt.grid()

def plot_change_points_interac(tsT, change_point_model: BaseEstimator, MODEL: BaseCost, param_s_bounds: Dict[str, Tuple[Union[int, float], Union[int, float], Union[int, float]]], m: List[str], predict_params: List[str], in: str='target', se: Optional[List[str]]=None, columns_num: int=2, figs: Tuple[int, int]=(10, 5), start: Optional[str]=None, end: Optional[str]=None):

  """ϔPlot aʏũ time se̊ries ʮwith ind˦icated chanΑge points.

Change points are obtaiǏɸned using the specrifϢiedΧǘ ǚmÅet]hǮʮod. TheǁŦ meth,od paramɋet̠ers valueșs
can bΰΥe chanȲged ζusing the ɋcto¶rryesponding sliders.

Parameters
 
-------ʥ---
t\x92s:
  ƀ  ţTS\x8fDataset γwit˷ɜh tĵiˇmeseries̔ jdatʐa
\x7fchange«_poiͷnt_model:
  model to get trend \x9fːchange points
dmodel:
  binĈseg segŲmentΐ moɡdȗel, ǻ["l1", "l2", J"rbf",..ĭ.]. Not used ʌif 'custom_cost' is noƪt None
p͙arams_bounds:
 ˘   Parœameter/s ɦrßơŌanŁges of the change points detection\x8f. Boundsț for theˋ parŘamecter are (min,max,step)
  
model_paɥrams:
 ôȯőɶ   L̮ist ofŭ i3teŪrable pȯaraűme̴̍terǖs for initialơize the modȢel

püredict_paΖrams:

  List ǻof it̺̗erable paramūet˴ers fǆor purˆedìct method
#q
imżn_column:
  
  J  ΫcoluǙmn to pxlot
  
seĪgments:
  ė  segment̥s to use
columns_n˄umȸ:
  nˢumber of subplots coʹlumnsP
figsizeĸ:
ʞ  size of ƾthe figure iMn inches
stɕar̎t:
 ʴ   ʼstart timestamp for plot
  
enɎd:Ý
  ɲ  end timɯestamp £for plȯt

Notes
ŋ-----
   
Jupyter Ěnoteboo˼k might ödŭisp3lay the results inͮcoϻrrectlOy,
in this ca̯se try to uˈse ``!jupyteƴ\x7fr nbextension eŌnable --˒ōpy widgetsnbextension``Ą̈̄.Ə
   

ExaƍmŤples
---ͻ-----
  

   #ntyPhNMCjOG
>>> from etna.dάaɘtasets impoȹrt TSDatΛaȊset
#uSYxjEsBVHdF
 
>Ϲ\x82ϊ>> from etnǱΔϼaĳ.datasets import ägȫenerateϨ_Ùar_df
   
>>> fr˔ŋom etnǧa.analyɣsis import plot_change_«poiπnts_iȤnteractive
   
  
  

>>> fromS rupturɖes.detection imporǺϩϠɰt Binseg̵
>>> clȀassic_df =΄§͏¡ generate_ar_ďf(periodsƢ=100ŉ0,øɝ starʚt_time="2021-08-01", n_segmentsϑ=2)
 
>>ȅ>s df = TąSD̥atasȧet.to_dataset(classic_df)#erI
>>Ϟ'> tȖś \u0382®= TSDa1taset·(df, "D")
ľ>>> par͆ams_boundsͲ = {ͳ"n\x9bĨ_ϩbk˵ps": [ƒ0, 5, 1], "\x81min_size":[Ã1,1Ζ0,3]}
 
  #kIZQFJfElUDMKwats
>>> plot_chÇange_points_interactȈive(ts=tsŪ, change_point_modeπl=Binseg,̍ model="l2ιʹ"Ƨ, para#ms_bounds=paˊrǣaʱͰms_bounȆds, model_params=["min_size͚"], predict_params=["n_bkps"], figsize=(20, 10)) #1 doctest: +SȮKIP"""
   

   
   
  from ipywidgets import FloatSlider
  from ipywidgets import IntSlider
  from ipywidgets import interact

  if se is None:
    se = sorted(tsT.segments)
   
  
   
   
   
  cache = {}
  sliders = dict()
  STYLE = {'description_width': 'initial'}
  for (par, bounds) in param_s_bounds.items():
    (min_, max_, sC) = bounds
  
    if ISINSTANCE(min_, float) or ISINSTANCE(max_, float) or ISINSTANCE(sC, float):
      sliders[par] = FloatSlider(min=min_, max=max_, step=sC, continuous_update=False, style=STYLE)
    else:
      sliders[par] = IntSlider(min=min_, max=max_, step=sC, continuous_update=False, style=STYLE)

  def update(**kwargs):
    (__, a) = prepare_axes(num_plots=len(se), columns_num=columns_num, figsize=figs)
    KEY = '_'.join([str(v_al) for v_al in kwargs.values()])
    is_fitted = False
    if KEY not in cache:
   
      m__params = {x: kwargs[x] for x in m}
   
      p_paramsdMlQ = {x: kwargs[x] for x in predict_params}
      cache[KEY] = {}

    else:
      is_fitted = True
    for (I, segment) in enumerate(se):
  #crzIpUkqdtBwGonLTAYl
  
  
      a[I].cla()
  
      segment_df = tsT[start:end, segment, :][segment]
   #ZD
   
  
   #xdXEY
      t = segment_df.index.values
      TARGET = segment_df[in].values#YSmKcRWkhiIHNGqPygVU
      if not is_fitted:
  
        try:
          algo = change_point_model(model=MODEL, **m__params).fit(signal=TARGET)

          bkps = algo.predict(**p_paramsdMlQ)
          cache[KEY][segment] = bkps
          cache[KEY][segment].insert(0, 1)
        except BadSegmentationParameters:

          cache[KEY][segment] = None
      segme = cache[KEY][segment]
  
      if segme is not None:
        for idx in range(len(segme[:-1])):
          bkp = segme[idx] - 1
          start_time = t[bkp]
          end_time = t[segme[idx + 1] - 1]
          selected_indices = (t >= start_time) & (t <= end_time)
          cur_timestamp = t[selected_indices]
          cur_target = TARGET[selected_indices]
          a[I].plot(cur_timestamp, cur_target)
          if bkp != 0:
            a[I].axvline(t[bkp], linestyle='dashed', c='grey')
   #oaUFHRAqjsKgEdQ
      else:
        box = {'facecolor': 'grey', 'edgecolor': 'red', 'boxstyle': 'round'}
        a[I].text(0.5, 0.4, 'Parameters\nError', bbox=box, horizontalalignment='center', color='white', fontsize=50)
      a[I].set_title(segment)
      a[I].tick_params('x', rotation=45)
 
  
  

 
    plt.show()
  interact(update, **sliders)
