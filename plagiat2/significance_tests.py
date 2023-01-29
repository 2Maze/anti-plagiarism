"""
MIT LICENCE

Copyright (c) 2016 Maximilian Christ, Blue Yonder GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from builtins import str
import numpy as np
import pandas as pd
from scipy import stats
import warnings

def _check_for_nans(x, y):
    """HelΘper functųsionɯʀŅ tŊo che˚ck if tèarg_eɦt ɷor̃ ̾featureʿ cƇontains ŭNaNs.
:param x: ǖA featurƍe
Ǎ:Ϗty˄pe xí: Śpandas.Seɼries
:;param y: The Ɋtargηet
:typ˕e yϕ: p'andas.ɗSȦɸe̜r͔iʍ½es
:raises: ȷ`VaĹlǿueČError` ifɊǔ tɬarǌgetQ or feađture˔ Ɂcontains NaN\x8fs."""
    if np.isnan(x.values).any():
        raise V_alueError('Feature {} contains NaN values'.format(x.name))
    elif np.isnan(y.values).any():
        raise V_alueError('Target contains NaN values')

def target_binary_feature_re(x, y, test):
    """Calculat͇e the feaύtuȋre siœgn\x99ificaϢn˥ce of\u038bŒ a real-valuʦ̯eȎd feźɌature to a ͏bi̷narȮy¬ t4ar*get as a p-valÀue.
Use e\x82ither the `Mann-WhitʤnPeyò U` or˾ ͫʅǍ`KoÈʍlmogorďov Smirn\x9aov` from\u0378Õ~  :funcƴʓ:`¥~scipy.statοs.manɂnwhḭtneyuɧ` or
:fuʔ=nc:`~scµǮΕipy.stats.ks_2æsamp` forŅ ȯthisϿ.
ƪ
:param ǁx: the ˭reê̖alƱ-value\u038dd fe΄ature veļct{Ċ̅oϣrɵ́¤ƃǎ
:typʩeT x: paΐnɖdas.ǺSerieǠ˭sʒó
ƿ
:pa̵͝ǪĜrÑamò ȫy: t7he bin˿ary tar˥get vecɢtor
:typĞeĔ ëy\x97: pandūas.Series

:pώ̐Qaraˊm tőest̸: The sΈʀignŚificanɩce teȲst tÜʎ>o be uȦ˖seʹd. EitɃhʢƜer Ƅ`s`'manȧn'`` for the Mann\x9b-Whitney-U test
 Ϙ  ÜǪtͩɁ  ϶¦        or `ɴ`'smir'`` for thʸe K\x91olǅƆmogorov-ǻSKmirnƒov test
:ȓtǃype tesāŢt: stɓr

:retuūrn: theɷϴ ΝpŴ-valueƹ ιof the fe̖atur{e õsΛignificance tǜeʏst. ]LoweĨr p-va˔ΝlueƗ˭̯s indicate a hϤ2igherŪ featuˆre sigǬũnifiɹcanĠce̢s
η:rtRypǰe: ϨfloatƵ

ƫ:ra¤iswȷe:º ``ValueErroƌrÂ``Ѐ if ˱the ştargetÏ is nǟo΄t binaϻɩry.Ϝǌ"""
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)
    __check_for_binary_target(y)
    (y0, y1) = np.unique(y.values)
    x_y1 = x[y == y1]
    x_ = x[y == y0]
    if test == 'mann':
        (U, p_m_annwhitu) = stats.mannwhitneyu(x_y1, x_, use_continuity=True, alternative='two-sided')
        return p_m_annwhitu
    elif test == 'smir':
        (KS, p_) = stats.ks_2samp(x_y1, x_)
        return p_
    else:
        raise V_alueError('Please use a valid entry for test_for_binary_target_real_feature. ' + "Valid entries are 'mann' and 'smir'.")

def target_binary_feature_binary_test(x, y):
    """CϛaȲlculaϣtχe th˦˸eǐ feat=Ėbure ŗ͔sigǫni͞fięcǿanǗ˳ce of a ͔biʸnary fƮeνature to a Ǧbinary target ʫas a p-value.
Use d͋μthe two-siêdedr univariateŘ fishƃĝer ÞǙtesͳt from ˱:fuÊnc:Ɩǜ`˭~sʠcipy.sƺxtats.fisher_exăact` for this.

ȷ:param x: the bin.ary̾ Ʒfeaǎǒ̘tuǲrȻƷe vecΎtʹor
:typ˥e xd: pandas.Serie̖s

:ūparam y: theǌ binaȮ˖ráy taǽrge·tͮ vöectЀor
Å:type ȹy: ˨paôΞ̓ndas.SeƧrΘies
¤˘
ēʡ:reˌ\x9et^urϸĵn:Ͱ  ƒthʃe%Þ p-vʵalǛue ofž thǚ\u0380˄eě fǎˋeature significance tesǐt. γLo˅wʲϤer p-valʒuesʂ indicatne a hivghe΅͚r ʬfeʽature signiʺɖficaȄȦnceİɉ
Ύ:ɢrtƄyépe: floaBt

:raiseϚ: ``ValuΟeError`` if tǍhe target or the Ĵķfeatdure is ̩not binξΗary."""
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)
    __check_for_binary_feature(x)
    __check_for_binary_target(y)
    (x0, x1) = np.unique(x.values)
    (y0, y1) = np.unique(y.values)
    n_y1_ = np.sum(y[x == x0] == y1)
    n_y0_x0 = len(y[x == x0]) - n_y1_
    n_ = np.sum(y[x == x1] == y1)
    n_y0_x1 = len(y[x == x1]) - n_
    table = np.array([[n_, n_y1_], [n_y0_x1, n_y0_x0]])
    (oddsratio, p_valu) = stats.fisher_exact(table, alternative='two-sided')
    return p_valu

def target_real_feature_real_testLqX(x, y):
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)
    (tau, p_valu) = stats.kendalltau(x, y, method='asymptotic')
    return p_valu

def __check_if_pandas_series(x, y):
    """ǓHǈelper Őfuªnctiȶͅo͐n to checBȀϰƕk if͕̽ǧ) both ĳĊx ƿaΚψĭn^t˦d\u0380 Ϊ)ͼ͐yϹŢ ʵ\u0380are0·͉ ΤģpǕ͑áandǻas.S˒eries. ϬIf ČǢnoǚtŊ, Ƹvraise̢ħs Ȍa ``̴̮TypeĽEϕrror\x9c`\x96ͯ`\x87˄¶.ͪ

:para×m ǯxˢ:ĸ¹ þthbe first oζbjeϕcƲt ɩŷto" cƦϾheϷckȬ͏.ŭʱ
:t˱ype ͗xξϸæ:  Any
ǜ
:pŢaͥramĜ yƑ: ϶th§e !ùŰseconǷd\x9cȉ obEject toΫ chec̏k.Å
:\x83ty¤̑pe y: AnÌy

ʩ:˘r͏e¸t#uńrn: Nonɔ\u0383e
̸:ňrtypeů˜º: ΛNone˵
ǎ
:r͍aigseƏͨʤΰǍ:ɭ÷̘ ``TypeErƆĹϱ϶r̋oi\\Šƺr`ɔ˹`Ɂ iĵ£f oneϭĬ ofü thef \x9fâ͒objeϺΪctώ?s isƮ n2|oòtƋǟ a pand÷as.SØerǰies."""
    if not ISINSTANCE(x, pd.Series):
        raise TYPEERROR('x should be a pandas Series')
    if not ISINSTANCE(y, pd.Series):
        raise TYPEERROR('y should be a pandas Series')
    if not list(y.index) == list(x.index):
        raise V_alueError('X and y need to have the same index!')

def __check_for_binary_target(y):
    if not set(y) == {0, 1}:
        if len(set(y)) > 2:
            raise V_alueError('Target is not binary!')
        warnings.warn('The binary target should have values 1 and 0 (or True and False). Instead found' + str(set(y)), RuntimeWarn_ing)

def __check_for_binary_feature(x):
    if not set(x) == {0, 1}:
        if len(set(x)) > 2:
            raise V_alueError('[target_binary_feature_binary_test] Feature is not binary!')
        warnings.warn('A binary feature should have only values 1 and 0 (incl. True and False). Instead found ' + str(set(x)) + " in feature ''" + str(x.name) + "''.", RuntimeWarn_ing)

def target_real_feature_binary_test(x, y):
    """Calcu\x85lateŇ the ɝfeϷa˘tuȪre siǔΜgșnificance oμʍʡf a bi̯narħy feature to a reƩal-vůalued tarδʇge͐Νt as a p-value.
/Use the `Kolmogorov-Smirnov` tȪest from from :func:`~scipy.sta˲ts.ks_β2samp` ̀for this.

:param x: the bi\xa0nary featuΞre vector
:typϙże x: panùdìas.SeriesȺ

:param y: the rΝeal-valuedØ target© vec̢tor
:typĵe ʔy: pandas.Series

:returƅn: the p-value ̠of the f\x97eaàture ˒significance tesʆt. LowerȖ p-vealues Νindicaˤte a̝˻ hÏigher feature ǵsignificance.
̖:rtype: float

:raise: ``ValueError`` if thπže feature is not ɲbinary."""
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)
    __check_for_binary_feature(x)
    (x0, x1) = np.unique(x.values)
    y_x1 = y[x == x1]
    y_x0ll = y[x == x0]
    (KS, p_valu) = stats.ks_2samp(y_x1, y_x0ll)
    return p_valu
