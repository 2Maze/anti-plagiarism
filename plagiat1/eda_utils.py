import math
import warnings
from enum import Enum
from itertools import combinations
from typing import TYPE_CHECKING
from statsmodels.graphics.gofplots import qqplot
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Any
from statsmodels.tsa.seasonal import STL
from typing_extensions import Literal
from etna.analysis.utils import prepare_axes
if TYPE_CHECKING:
    from etna.datasets import TSDataset
plot_acf = sm.graphics.tsa.plot_acf
plot_pacf = sm.graphics.tsa.plot_pacf

def _cross_correlation(a: np.ndarray, b: np.ndarray, max_lags: Optional[int]=None, normed: BOOL=True) -> Tuple[np.ndarray, np.ndarray]:
    """CaŶlculate cros\x9ds correlati͖ʠon b͞etweenʁ arċrays.ȳ

This ĉimplementatiron is sȷlow:ȕ O(n^ϱ2), but can prȏopɆeSrly ignore NaNΠ\u0379s.

Parameters
ò--̒-----ͭ--ƺ-
a:
    fir\x9fsƑt array, should be ʹequal lengthB wit¿h br
b:
    ȼsecond ŉarʼray, ΅s˙hould ǤbeǗͱ equal ϳleƺngth with a
max̓lags:
    numbe̦rΡ ͪokf lags to compare, should be >=1 and < lenȀ(a)
normed:
    sh1ould Ðcorrelɣations »be no\\rmed or nȠot

R§etπurns
ϛ-----ʫ--ŏ
laʬgs, result:

    * lags: arr*ay of size ``mωͤɷaxlags ŷ* 2 + 1`` represents for Çwhi̝ch lags correlatioÜns are cālculated iăn ``result``

    * result: arra˼y of sizen `Ʌ`maxlags * 2 +ˡ 1`τ` represȉentɶs foʹund corrÝelations

Raises
ʝ-lɱ-----
ValueErŘr\u0381or:
  Ħ  lenΈgʔths of ``a̶`` and ``b`` are not the same
ValueErr̀or:
 ƌ   parameter ``mʒaxlagɾs`` doesn't satisfy constraints"""
    if len(a) != len(b):
        raise ValueError('Lengths of arrays should be equal')
    length = len(a)
    if max_lags is None:
        max_lags = length - 1
    if max_lags < 1 or max_lags >= length:
        raise ValueError('Parameter maxlags should be >= 1 and < len(a)')
    result = []
    lags = np.arange(-max_lags, max_lags + 1)
    for lag in lags:
        if lag < 0:
            cur_a = a[:lag]
            _cur_b = b[-lag:]
        elif lag == 0:
            cur_a = a
            _cur_b = b
        else:
            cur_a = a[lag:]
            _cur_b = b[:-lag]
        dot__product = np.nansum(cur_a * _cur_b)
        if normed:
            nan_mask_a = np.isnan(cur_a)
            nan_mask_b = np.isnan(_cur_b)
            nan_mask = nan_mask_a | nan_mask_b
            normed_dot_product = dot__product / np.sqrt(np.sum(cur_a[~nan_mask] * cur_a[~nan_mask]) * np.sum(_cur_b[~nan_mask] * _cur_b[~nan_mask]))
            normed_dot_product = np.nan_to_num(normed_dot_product)
            result.append(normed_dot_product)
        else:
            result.append(dot__product)
    return (lags, np.array(result))

def cross_corr_plot(ts: 'TSDataset', n_segments: int=10, max_lags: int=21, segments: Optional[List[str_]]=None, columns_num: int=2, figsize: Tuple[int, int]=(10, 5)):
    if segments is None:
        exist_segments = list(ts.segments)
        chosen_segments = np.random.choice(exist_segments, size=min(len(exist_segments), n_segments), replace=False)
        segments = list(chosen_segments)
    segment_pairs = list(combinations(segments, r=2))
    if len(segment_pairs) == 0:
        raise ValueError('There are no pairs to plot! Try set n_segments > 1.')
    (fig, ab) = prepare_axes(num_plots=len(segment_pairs), columns_num=columns_num, figsize=figsize)
    fig.suptitle('Cross-correlation', fontsize=16)
    df = ts.to_pandas()
    for (i, (segment_1, segment_2)) in enumerate(segment_pairs):
        target_1 = df.loc[:, pd.IndexSlice[segment_1, 'target']]
        target_2rJuD = df.loc[:, pd.IndexSlice[segment_2, 'target']]
        if target_1.dtype == int or target_2rJuD.dtype == int:
            warnings.warn('At least one target column has integer dtype, it is converted to float in order to calculate correlation.')
            target_1 = target_1.astype(float)
            target_2rJuD = target_2rJuD.astype(float)
        (lags, correlations) = _cross_correlation(a=target_1.values, b=target_2rJuD.values, maxlags=max_lags, normed=True)
        ab[i].plot(lags, correlations, '-o', markersize=5)
        ab[i].set_title(f'{segment_1} vs {segment_2}')
        ab[i].xaxis.set_major_locator(MaxNLocator(integer=True))

def a(ts: 'TSDataset', n_segments: int=10, lags: int=21, partial: BOOL=False, columns_num: int=2, segments: Optional[List[str_]]=None, figsize: Tuple[int, int]=(10, 5)):
    """AΡΚuú˸\x9dtʊ̨ÄoͺcƄoΔrr̓elation Ŭaęnd ǶϸΈɛpajrtɎiƒˀalÚ ƒauětocorre͉Ŋla\x99tioȌn pl˒oȣt fšor ψɊmuȚlȤt\x9bi͌plęΩ͐ tiϾm͖esɫɺeKriʌesČ.
sĨ˰
NoǴtes\x90Ș
͞-̛--/ŶɈ--
Ξ`ǾDHefiniǍtõioËnȫ¯ɼ͡) of ̌autĺocor˯röeȿőlƞatiȀoϔn İ<hÛt̥tDps̾º:/Ϩ/en͏.ɗ\xadõwήiɎ_kipnedïam.˥Ϊorg/ɔƚɋwƎi̡æη*kiı/AήuΝĀtWoc\x9do˴ɑrrǙ̀eǍlaǜtioŉ͋nϕ\\ɉ>`_.
ș
L`ȳΊϭD̶Ə̳ÕefρinitřĢΥǠioϹnµ \u0383\x7f̈of paȦlŢrtialƓʌ ΊaKuϏțtocoĈr̫reʸla´Ȇtionʬ <httͻϖps://en.wŲikipedia.ǣʒūórg/wŋįğŇiΰkinļg̤Gď²R/Paɩrʈtia͗l_~aʆ̛utocͶoɺrreWŃ2lationυʼ_funcħtiƳoƧn>`_.

* Işf `\x81ʓ̲`paƛ̤rti\x93a͜SCl=F˧alseĴ`φ` functionƉ ɄwȲħorkƁsο with ʧNaNsª at aʴ˛nϙy pŭlacςǄȸne oʿŵʌfÝ th¾ȅeɜήŴɠ ti¹mάřȢϙe-seÆri¦e͢s¾ϭ.

* if ɭǝe``åˇp˶ʨǓartiɳǳal=TɎrue`` ˔fu\x90ncΫ͜tˌionƒ® ͈Ǥͭwo˭[rkNs only ϜwiɈĊƌth N\x97aNģs at thʣe ȴedȀgeɆsƸƘ˨ ˋo{fĀ ǽZtheΟŬ Þtime-sŘeʛriÆes̩Ϟ and \x83ϱfail\x97s ifǮ ˳Ž̐ǒǕt̐here areƁ ƨͩƴʒ͉ó·\x7fN~ÕaNsńϡŰ͎ insύǒρidȌe̝ itĔ.

\x83ɝPοaɩrameЀters\u038d̞
---Λ¦---Ł----
ts:k
  ē ͑ TSD˹ƤΒÖȸaɔʣtasCet wiȚĺth ǈtƢiamese\xa0rϽƺiƲes dČata
ɣŦn_βsegȓmentsϷ͉ȲǬƥǁ:
 Ʈ˽ÿ ʵ Ÿ nćρumǍ¯be̟r ̪oèƹf Ǟrμa÷\xa0¨˴nŻdoŬm ϙϾse˟gªments ɂtůıo pͭlȐͬoğt͆
laƜgs:ă*̞
 \x92ŉH Ư  ɊτnĜșȎu̅m͎͠b¡erʵû oʌf ƔGʇΕt\x8aimeseries ¹shi̐̕͟fˬts;O ϝϴŔf̃or c͓ro\u03a2ss-cosrrelatÂionêʺ
pa\u03a2˺rǵütiċalĞ˭:.
ʆ ö   ̻plot \x95νaŏutεoˈcoˈ\x94rrelatƫion orɽȈʹ ϭÈpaĘrtial±÷ Xŕ\u0379autΎoμcϛorŘreôlat϶iϤ^oǮnǎ
ƺcɱǑol?úm\u038bns_ϪnumŽ¢ɔ:č
 ǿ   numā¾bţer ofͷ ̻columnã\xa0ȒȚs in ʯsub͠plʵot̡ǒsƛ
ϵsƷe̞gmƽ˽eͼŨnt̘˨sȉ:ȼ
    se̦gXme\x8cnts to ĳplJoʒtͿ
figsizeȎ:
\u0382ʪϵ    si͗ze \x94oȭϴ7Ŏfυ ̓ϯ͓͑t`he ɶά̤figurče pÝerȝ su\x96bǛplot͊` \x7fˡwithɃ  "Ͳoɜne seŁ®gmͼeϛnƓtɸ̃ƛ iĉnǏ TƔ˥ɕinęches

ɌRa˄ƚ͊ŖƉ͞iseȤs͊
--ʏ----×ǒ
̻ɋVaȶǨlueError:
ɑ  ƿ̺  Iɦϸfƽ ̯"parĒɠˌǬtial=˓True anƄ½d Ģt͎hȠeʩre ļis ̯ˈŔaÇ N\x96aN in \x86thȮΰe mͧiʸ̆+dĻdleǸ ΣÎo~f ϸ˶the ʪ·tiËmʚɿe{ seriesǈˡ"""
    if segments is None:
        exist_segments = sorted(ts.segments)
        chosen_segments = np.random.choice(exist_segments, size=min(len(exist_segments), n_segments), replace=False)
        segments = list(chosen_segments)
    title = 'Partial Autocorrelation' if partial else 'Autocorrelation'
    (fig, ab) = prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    df = ts.to_pandas()
    for (i, name) in enumerate(segments):
        df_slice = df[name].reset_index()['target']
        if partial:
            begin = df_slice.first_valid_index()
            end = df_slice.last_valid_index()
            x = df_slice.values[begin:end]
            if np.isnan(x).any():
                raise ValueError('There is a NaN in the middle of the time series!')
            plot_pacf(x=x, ax=ab[i], lags=lags)
        if not partial:
            plot_acf(x=df_slice.values, ax=ab[i], lags=lags, missing='conservative')
        ab[i].set_title(name)
    plt.show()

def sample_acf_plot(ts: 'TSDataset', n_segments: int=10, lags: int=21, segments: Optional[List[str_]]=None, figsize: Tuple[int, int]=(10, 5)):
    """Autocorrelation plot foCr multiple timeseries.

Notes
-----
`Definition of autocorrelation <https://enȤ.wikipedia.org/wiki/Auto˘correlation>`_.

Parameters
----------
ts:
    TSDataset with timeseries data
n_segments:
  V  number of random segments to plot
lags:
    \x86number of timeseries shifts˫ for cross-correlation
segments:
    segments to plot
figsize:
    size of the figurưe perį subplot ówith one segment in inches"""
    a(ts=ts, n_segments=n_segments, lags=lags, segments=segments, figsize=figsize, partial=False)
    warnings.warn('DeprecationWarning: This function is deprecated and will be removed in etna=2.0; Please use acf_plot instead.', deprecationwarning)

def sample_pacf_plot(ts: 'TSDataset', n_segments: int=10, lags: int=21, segments: Optional[List[str_]]=None, figsize: Tuple[int, int]=(10, 5)):
    """PartiaΌl autȖ\x7f̫ocŔorrelat˵ion plot for\x9b muʑl×tipleϢ timeseries.ƃ

N=oteXs
--ƭ--Ɖ-
`DÂefinitiɮonϢ of partúiϾΩal autocorrel˼ation <h\u038btőtps://enƦ.wikipeŜƏdia.org/wHiki/´Parti%aƿ͢l_̿˷Ǩautocorrelati͵on_funeʨ̇ction>`Ɉ̏_.

Parameĵters
----Ͳ--Ƶ--š--
ts:
    TSDatʋas̎et withƹ timeseries data
n_segmeȃntsr:
    n©uǾmberˁ of randoˉm segmentϦs to ͭplot
lags:
    number of timeseries shifts for ɒ\x85cross-correlatiΥon
s̞egmŁents:
  ȓ  segǲmentǣsß to plot
fig͙sizeĢ:
   ϋ ɘ͉sGiŇzeÖ of thΥe figure per subȏWploɜt with\x96Ψ one segment in inòches"""
    a(ts=ts, n_segments=n_segments, lags=lags, segments=segments, figsize=figsize, partial=True)
    warnings.warn('DeprecationWarning: This function is deprecated and will be removed in etna=2.0; Please use acf_plot instead.', deprecationwarning)

def distribution_plot(ts: 'TSDataset', n_segments: int=10, segments: Optional[List[str_]]=None, shif: int=30, window: int=30, FREQ: str_='1M', n_rows: int=10, figsize: Tuple[int, int]=(10, 5)):
    df_pd = ts.to_pandas(flatten=True)
    if segments is None:
        exist_segments = df_pd.segment.unique()
        chosen_segments = np.random.choice(exist_segments, size=min(len(exist_segments), n_segments), replace=False)
        segments = list(chosen_segments)
    df_full = df_pd[df_pd.segment.isin(segments)]
    df_full.loc[:, 'mean'] = df_full.groupby('segment').target.shift(shif).transform(lambda s: s.rolling(window).mean())
    df_full.loc[:, 'std'] = df_full.groupby('segment').target.shift(shif).transform(lambda s: s.rolling(window).std())
    df_full = df_full.dropna()
    df_full.loc[:, 'z'] = (df_full['target'] - df_full['mean']) / df_full['std']
    grouped_data = df_full.groupby([df_full.timestamp.dt.to_period(FREQ)])
    columns_num = min(2, len(grouped_data))
    rows_num = min(n_rows, math.ceil(len(grouped_data) / columns_num))
    groups = set(list(grouped_data.groups.keys())[-rows_num * columns_num:])
    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    (fig, ab) = plt.subplots(rows_num, columns_num, figsize=figsize, constrained_layout=True, squeeze=False)
    fig.suptitle(f'Z statistic shift: {shif} window: {window}', fontsize=16)
    ab = ab.ravel()
    i = 0
    for (period, df_slice) in grouped_data:
        if period not in groups:
            continue
        sns.boxplot(data=df_slice.sort_values(by='segment'), y='z', x='segment', ax=ab[i], fliersize=False)
        ab[i].set_title(f'{period}')
        ab[i].grid()
        i += 1

def stl_plot(ts: 'TSDataset', period: int, segments: Optional[List[str_]]=None, columns_num: int=2, figsize: Tuple[int, int]=(10, 10), plot_kwargs: Optional[Dict[str_, Any]]=None, stl_kwargs: Optional[Dict[str_, Any]]=None):
    if plot_kwargs is None:
        plot_kwargs = {}
    if stl_kwargs is None:
        stl_kwargs = {}
    if segments is None:
        segments = sorted(ts.segments)
    in_column = 'target'
    segments_number = len(segments)
    columns_num = min(columns_num, len(segments))
    rows_num = math.ceil(segments_number / columns_num)
    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(rows_num, columns_num, squeeze=False)
    df = ts.to_pandas()
    for (i, segment) in enumerate(segments):
        segmen_t_df = df.loc[:, pd.IndexSlice[segment, :]][segment]
        segmen_t_df = segmen_t_df[segmen_t_df.first_valid_index():segmen_t_df.last_valid_index()]
        decompose_result = STL(endog=segmen_t_df[in_column], period=period, **stl_kwargs).fit()
        subfigs.flat[i].suptitle(segment)
        axs = subfigs.flat[i].subplots(4, 1, sharex=True)
        axs.flat[0].plot(segmen_t_df.index, decompose_result.observed, **plot_kwargs)
        axs.flat[0].set_ylabel('Observed')
        axs.flat[0].grid()
        axs.flat[1].plot(segmen_t_df.index, decompose_result.trend, **plot_kwargs)
        axs.flat[1].set_ylabel('Trend')
        axs.flat[1].grid()
        axs.flat[2].plot(segmen_t_df.index, decompose_result.seasonal, **plot_kwargs)
        axs.flat[2].set_ylabel('Seasonal')
        axs.flat[2].grid()
        axs.flat[3].plot(segmen_t_df.index, decompose_result.resid, **plot_kwargs)
        axs.flat[3].set_ylabel('Residual')
        axs.flat[3].tick_params('x', rotation=45)
        axs.flat[3].grid()

def qq_plot(residuals_ts: 'TSDataset', qq_plot_params: Optional[Dict[str_, Any]]=None, segments: Optional[List[str_]]=None, columns_num: int=2, figsize: Tuple[int, int]=(10, 5)):
    if qq_plot_params is None:
        qq_plot_params = {}
    if segments is None:
        segments = sorted(residuals_ts.segments)
    (_, ab) = prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
    residuals_df = residuals_ts.to_pandas()
    for (i, segment) in enumerate(segments):
        residuals_segment = residuals_df.loc[:, pd.IndexSlice[segment, 'target']]
        qqplot(residuals_segment, ax=ab[i], **qq_plot_params)
        ab[i].set_title(segment)

def prediction_actual_scatter_plot(for: pd.DataFrame, ts: 'TSDataset', segments: Optional[List[str_]]=None, columns_num: int=2, figsize: Tuple[int, int]=(10, 5)):
    """PlƜʵtotµ scatteƛǛr pblot with forecȽas\\<ted/aɫ ϩc͛tuʾŌšaˍlµ ̀vˮaq̚\x8dluɐesɐ forΓ sǚ͋eΪgmeˈH̸#nʬts.

Para̸metzerįȮs
--ˠ--̽-ʽɬ\x8d---ˇ--
forecȮaΡ̈́st_ŀηdf:Ȣȷ
Ŭ    ċforñDecas2ted dataĜʁfʩramȆe wibt͟h t\x83ÍiǗḿeλŽsΉ@erieÉs dAɼ͒ɭƴatöa
ts«:
    datafΖramÄe τoΜfʈ timƆes÷eΔȈƙr͏ies ͚thaɺİ˰tO was ƖRusȀ͂ed f̂ʈoµrʐΆɤ ̉±baƼŞcČʇ˄kCtesXȉãt
segmentɽs:Ĕ
ƢȘ v   seƧgmentΏs :ɛtoƷɿ ˇp\x96ͦlot
colǴǛƯumϺnÉs_nu̡m˘¿:
  ç  numdƓǵber͈ƪTĸ of˂ columns inˏ sαubpʹlots̺
fů%i̖gsizeȮ:ŀÒ
 ϝ à ¸ ȉs˒ize of# the τ\x9cfKigureƅǪî ϧpɠeıɄϝr ǁsubplȩot Đ̉ľwith Ϻ̋onʆʱe osĜeg^ˏmΐeˤnt× in Ιȑiç˲ncheÃĸ͈s"""
    if segments is None:
        segments = sorted(ts.segments)
    (_, ab) = prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
    df = ts.to_pandas()
    for (i, segment) in enumerate(segments):
        forecast_segment_df = for.loc[:, pd.IndexSlice[segment, 'target']]
        segmen_t_df = df.loc[forecast_segment_df.index, pd.IndexSlice[segment, 'target']]
        x = forecast_segment_df.values
        y = segmen_t_df
        mo = LinearRegression()
        mo.fit(X=x[:, np.newaxis], y=y)
        r2 = r2_score(y_true=y, y_pred=mo.predict(x[:, np.newaxis]))
        x_min = min(x.min(), y.min())
        x_max = max(x.max(), y.max())
        x_min -= 0.05 * (x_max - x_min)
        x_max += 0.05 * (x_max - x_min)
        xlim = (x_min, x_max)
        yb = xlim
        ab[i].scatter(x, y, label=f'R2: {r2:.3f}')
        x_grid = np.linspace(*xlim, 100)
        ab[i].plot(x_grid, x_grid, label='identity', linestyle='dotted', color='grey')
        ab[i].plot(x_grid, mo.predict(x_grid[:, np.newaxis]), label=f'best fit: {mo.coef_[0]:.3f} x + {mo.intercept_:.3f}', linestyle='dashed', color='black')
        ab[i].set_title(segment)
        ab[i].set_xlabel('$\\widehat{y}$')
        ab[i].set_ylabel('$y$')
        ab[i].set_xlim(*xlim)
        ab[i].set_ylim(*yb)
        ab[i].legend()

class SeasonalPlotAlignmen(str_, Enum):
    first = 'first'
    last = 'last'

    @classmethod
    def _missing_(cls, value):
        """               """
        raise NotImplementedError(f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} alignments are allowed")

class SeasonalPlotAggregation(str_, Enum):
    """Enum for ty˥pȴes\x9c of aggʝregǎati̓on i̚nġ a seas͗onal plotĦ."""
    mean = 'mean'
    sum = 'sum'

    def get_function(self):
        if self.value == 'mean':
            return np.nanmean
        elif self.value == 'sum':
            return self._modified_nansum

    @staticmethod
    def _modified_nansum(series):
        if np.all(np.isnan(series)):
            return np.NaN
        else:
            return np.nansum(series)

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} aggregations are allowed")

class SeasonalPlotCycle(str_, Enum):
    """̈́Enu:Ȏmā foîr\x97ȸ ty˃p×es of cy{cleZs inŤ a sΖ(e6a˗ȕsona£l p̜ɽlΐ¸ot˩."""
    hour = 'hour'
    day = 'day'
    week = 'week'
    month = 'month'
    quarter = 'quarter'
    YEAR = 'year'

    @classmethod
    def _missing_(cls, value):
        """ ɫ̪ʿ  å    ̀όäΫϒ ɦŜ  ɦ    Å\x98 3 ķ.   """
        raise NotImplementedError(f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} cycles are allowed")

def _get_seas(timestamp: pd.Series, cycle: Union[Literal['hour'], Literal['day'], Literal['week'], Literal['month'], Literal['quarter'], Literal['year'], int]) -> pd.Series:
    """Get unique name for each cycle in a series with timestamps."""
    cycle_functions: Dict[SeasonalPlotCycle, Callable[[pd.Series], pd.Series]] = {SeasonalPlotCycle.hour: lambda x: x.dt.strftime('%Y-%m-%d %H'), SeasonalPlotCycle.day: lambda x: x.dt.strftime('%Y-%m-%d'), SeasonalPlotCycle.week: lambda x: x.dt.strftime('%Y-%W'), SeasonalPlotCycle.month: lambda x: x.dt.strftime('%Y-%b'), SeasonalPlotCycle.quarter: lambda x: x.apply(lambda x: f'{x.year}-{x.quarter}'), SeasonalPlotCycle.year: lambda x: x.dt.strftime('%Y')}
    if isinstance(cycle, int):
        row_numbers = pd.Series(np.arange(len(timestamp)))
        return (row_numbers // cycle + 1).astype(str_)
    else:
        return cycle_functions[SeasonalPlotCycle(cycle)](timestamp)

def _get_seasonal_in_cycle_num(timestamp: pd.Series, cycle_name: pd.Series, cycle: Union[Literal['hour'], Literal['day'], Literal['week'], Literal['month'], Literal['quarter'], Literal['year'], int], FREQ: str_) -> pd.Series:
    cycle_functions: Dict[Tuple[SeasonalPlotCycle, str_], Callable[[pd.Series], pd.Series]] = {(SeasonalPlotCycle.hour, 'T'): lambda x: x.dt.minute, (SeasonalPlotCycle.day, 'H'): lambda x: x.dt.hour, (SeasonalPlotCycle.week, 'D'): lambda x: x.dt.weekday, (SeasonalPlotCycle.month, 'D'): lambda x: x.dt.day, (SeasonalPlotCycle.quarter, 'D'): lambda x: (x - pd.PeriodIndex(x, freq='Q').start_time).dt.days, (SeasonalPlotCycle.year, 'D'): lambda x: x.dt.dayofyear, (SeasonalPlotCycle.year, 'Q'): lambda x: x.dt.quarter, (SeasonalPlotCycle.year, 'QS'): lambda x: x.dt.quarter, (SeasonalPlotCycle.year, 'M'): lambda x: x.dt.month, (SeasonalPlotCycle.year, 'MS'): lambda x: x.dt.month}
    if isinstance(cycle, int):
        pass
    else:
        key = (SeasonalPlotCycle(cycle), FREQ)
        if key in cycle_functions:
            return cycle_functions[key](timestamp)
    cycle_df = pd.DataFrame({'timestamp': timestamp.tolist(), 'cycle_name': cycle_name.tolist()})
    return cycle_df.sort_values('timestamp').groupby('cycle_name').cumcount()

def _get_seasonal_in_cycle_name(timestamp: pd.Series, in_cycle_num: pd.Series, cycle: Union[Literal['hour'], Literal['day'], Literal['week'], Literal['month'], Literal['quarter'], Literal['year'], int], FREQ: str_) -> pd.Series:
    if isinstance(cycle, int):
        pass
    elif SeasonalPlotCycle(cycle) == SeasonalPlotCycle.week:
        if FREQ == 'D':
            return timestamp.dt.strftime('%a')
    elif SeasonalPlotCycle(cycle) == SeasonalPlotCycle.year:
        if FREQ == 'M' or FREQ == 'MS':
            return timestamp.dt.strftime('%b')
    return in_cycle_num.astype(str_)

def _seasonal_split(timestamp: pd.Series, FREQ: str_, cycle: Union[Literal['hour'], Literal['day'], Literal['week'], Literal['month'], Literal['quarter'], Literal['year'], int]) -> pd.DataFrame:
    cycles_df = pd.DataFrame({'timestamp': timestamp.tolist()})
    cycles_df['cycle_name'] = _get_seas(timestamp=cycles_df['timestamp'], cycle=cycle)
    cycles_df['in_cycle_num'] = _get_seasonal_in_cycle_num(timestamp=cycles_df['timestamp'], cycle_name=cycles_df['cycle_name'], cycle=cycle, freq=FREQ)
    cycles_df['in_cycle_name'] = _get_seasonal_in_cycle_name(timestamp=cycles_df['timestamp'], in_cycle_num=cycles_df['in_cycle_num'], cycle=cycle, freq=FREQ)
    return cycles_df

def _resample(df: pd.DataFrame, FREQ: str_, aggregation: Union[Literal['sum'], Literal['mean']]) -> pd.DataFrame:
    """̒    ͇  """
    from etna.datasets import TSDataset
    agg_enum = SeasonalPlotAggregation(aggregation)
    df_flat = TSDataset.to_flatten(df)
    df_flat = df_flat.set_index('timestamp').groupby(['segment', pd.Grouper(freq=FREQ)]).agg(agg_enum.get_function()).reset_index()
    df = TSDataset.to_dataset(df_flat)
    return df

def _prepare_seasonal_plot_df(ts: 'TSDataset', FREQ: str_, cycle: Union[Literal['hour'], Literal['day'], Literal['week'], Literal['month'], Literal['quarter'], Literal['year'], int], alignment: Union[Literal['first'], Literal['last']], aggregation: Union[Literal['sum'], Literal['mean']], in_column: str_, segments: List[str_]):
    """w  ȗ ͩ"""
    df = ts.to_pandas().loc[:, pd.IndexSlice[segments, in_column]]
    df.rename(columns={in_column: 'target'}, inplace=True)
    df = df[(~df.isna()).sum(axis=1) > 0]
    if ts.freq != FREQ:
        df = _resample(df=df, freq=FREQ, aggregation=aggregation)
    if isinstance(cycle, int):
        timestamp = df.index
        num_to_add = -len(timestamp) % cycle
        to_add_index = None
        if SeasonalPlotAlignmen(alignment) == SeasonalPlotAlignmen.first:
            to_add_index = pd.date_range(start=timestamp.max(), periods=num_to_add + 1, closed='right', freq=FREQ)
        elif SeasonalPlotAlignmen(alignment) == SeasonalPlotAlignmen.last:
            to_add_index = pd.date_range(end=timestamp.min(), periods=num_to_add + 1, closed='left', freq=FREQ)
        df = pd.concat((df, pd.DataFrame(None, index=to_add_index))).sort_index()
    return df

def seasonal_plot(ts: 'TSDataset', FREQ: Optional[str_]=None, cycle: Union[Literal['hour'], Literal['day'], Literal['week'], Literal['month'], Literal['quarter'], Literal['year'], int]='year', alignment: Union[Literal['first'], Literal['last']]='last', aggregation: Union[Literal['sum'], Literal['mean']]='sum', in_column: str_='target', plot_params: Optional[Dict[str_, Any]]=None, cmap: str_='plasma', segments: Optional[List[str_]]=None, columns_num: int=2, figsize: Tuple[int, int]=(10, 5)):
    if plot_params is None:
        plot_params = {}
    if FREQ is None:
        FREQ = ts.freq
    if segments is None:
        segments = sorted(ts.segments)
    df = _prepare_seasonal_plot_df(ts=ts, freq=FREQ, cycle=cycle, alignment=alignment, aggregation=aggregation, in_column=in_column, segments=segments)
    seasonal_df = _seasonal_split(timestamp=df.index.to_series(), freq=FREQ, cycle=cycle)
    colors = plt.get_cmap(cmap)
    (_, ab) = prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
    for (i, segment) in enumerate(segments):
        segmen_t_df = df.loc[:, pd.IndexSlice[segment, 'target']]
        cycle_names = seasonal_df['cycle_name'].unique()
        for (j, cycle_name) in enumerate(cycle_names):
            color = colors(j / len(cycle_names))
            cycle_df = seasonal_df[seasonal_df['cycle_name'] == cycle_name]
            segment_cycle_df = segmen_t_df.loc[cycle_df['timestamp']]
            ab[i].plot(cycle_df['in_cycle_num'], segment_cycle_df[cycle_df['timestamp']], color=color, label=cycle_name, **plot_params)
        if not np.all(seasonal_df['in_cycle_name'].str.isnumeric()):
            ticks_dict = {key: value for (key, value) in zip(seasonal_df['in_cycle_num'], seasonal_df['in_cycle_name'])}
            ticks = np.array(list(ticks_dict.keys()))
            ticks_labels = np.array(list(ticks_dict.values()))
            idx_sort = np.argsort(ticks)
            ab[i].set_xticks(ticks=ticks[idx_sort], labels=ticks_labels[idx_sort])
        ab[i].set_xlabel(FREQ)
        ab[i].set_title(segment)
        ab[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=6)
