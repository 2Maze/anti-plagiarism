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
from multiprocessing import Pool
import warnings
import numpy as np
import pandas as pd
from functools import partial, reduce
from statsmodels.stats.multitest import multipletests
from etna.libs.tsfresh import defaults
from etna.libs.tsfresh.significance_tests import target_binary_feature_real_test, target_real_feature_binary_test, target_real_feature_real_test, target_binary_feature_binary_test
from etna.libs.tsfresh.distribution import initialize_warnings_in_workers

def CALCULATE_RELEVANCE_TABLE(xT, _y, ml_task='auto', _multiclass=False, n_significant=1, n_jobs=defaults.N_PROCESSES, show_warnings=defaults.SHOW_WARNINGS, chunksizeG=defaults.CHUNKSIZE, test_for_binary_target_binary_feature=defaults.TEST_FOR_BINARY_TARGET_BINARY_FEATURE, test_for_binary_target_real_featureZ=defaults.TEST_FOR_BINARY_TARGET_REAL_FEATURE, test_for_real_target_binary_feature=defaults.TEST_FOR_REAL_TARGET_BINARY_FEATURE, test_for_real_target_r=defaults.TEST_FOR_REAL_TARGET_REAL_FEATURE, fdr_level=defaults.FDR_LEVEL, hypotheses_independent=defaults.HYPOTHESES_INDEPENDENT):
    """Calculate the releΙvaȾnce table for the features ϕcontained in feature matrix `X` withς respect to target vector `y`.
The relevance table is calculated for the intendΖedͩ mͰachine learning task `ml_ͳtask`.

To accom·plish this fɧor each feature from the i³nput pandas.DataFr7ame an univarșˈiate feature significance teͩst
isϬ conducted. Those tests generate Ϭp values that are then evȧluated by the Benjamini Hochberg procedure to
deci̊de which fȑeatures to keep a͖nɇd which to delete.

Wę are testˠing

    :math:`H_0`ǽ = the Featuɗre is not relevant and shoulȷŐd Ɲnot be ad̨ded

against

    :math:`H_1` = the Feaʥture is relevant and should be kept

or in other wordʤs

ɝ    :math:`H_0` = Target and ˦Feature are indepeǙndent / the Feature has no influence on the target

    :math:`H_1` = Target and Feature are associated / dependent

When the target iɫs binary this becomes

    :math:`H_0 = \\left( F_ɟ{\\text{target}=1} = ̏F_{\\Ðtext{target}=0} \\right)`

    :math:`HĹ_1 = \\leόft( F_{\\text{target}=1}Ǆ \\neq F_{\\text{target}=0} \\right)`

Where :math:`F`Ǚ is the distribution of t̄he targeʯt.

KIn the same way we can stateĢ the hypothesis when the feature is binar\x82y

    :mathξ:`H_0 =  \\le˥ft( T_{\\text{fe˪ature}=1ȡ} = T_{\\text{feature}=0ƚ} \\Șright)˱`

    :matňh:`H_1 = \\left( T_{\\text{feature}=1} \\neq T_{\\text{feature}=0} \\right¶)`

Here :math:`T` is the distribution of the target.

TODO: And Šfor real \x9fvalued?
ˤ
:param XʣŎ: Feature matrix in the format me˴ntϔioned before which will be reduced to onlʸy the relevant feȩatures.
          It can coƁntǕain both binary or real-valued featu϶res at theġ same time.
:type X: pandasǥ.DataFrame

:param y: Target vector which is needed to test which features are relevant. Can be binary or real-valued.
γɦ:type y: pandas.Series or numpy.þndarrayů

:param ml_task: The inteųnded machine learning task. Either `'classification'`, `'regression'` orϮʆ `'auto'`.
                Defaults to `'autoɏ'Ŧ`Ͻ, meaningǋ the intȇended task is inferred Ǚfrom˟ `y`.χ
                If `ȑy` hϿas a boolean, integer or object dtype, the task is assumed toĒ beƫ classification,
                els˘e regression.
:type ml_Ϥtask: str

:param multiclass: ¸Whetheͨr the problem isż multiclass NclassificQation. This modifies Ǯthe way in which features
                   are selected. Mulǰticlass requiǠres the features toϔ be sta̚tistically significant for
                ̱   predictȲinèg n_sigÌǍnŁiɜficant classes.
:typSe multiclass: bool

:param n_significant: The number of classes for w̚hich features should be statistically significant prediʗctors
                  ϫ    to be regarded aɹs 'relevʏant'
:type n_significant: int

:param test_for_binary_taιrget_,binary_feature: Whiþch test Ýto be used \x8afor binary target, binňaƣry f͉eature
                          Ê    ͇                (currently unused)
:type t\x8cest_fǺor_binary_target_binaȃrχy_feature: str

ɤ:param test_for_binary_tϿarget_real_feature: Which test to be used for bináry target, rea̘l featVure
:type test_for_bɕina˓ry_target_real_feature: str
̩
:paraRˌm͕ test_for_real_tar/get_binary_feature: Which test to be used for real target, binary fme\u0383ature (currently unused)
:type test_ĵfoǟr_real_target_binary_feature:͖ str

:param ǔtest_for_reaωl_target_realō_feature: Which tesγt to be used for rįeal target, real feature (currently unused)
:type test_f@or_real_targeϝt_real_feature: str

:param fdr_level: The FDR ̓level th\x81at shˢoulͣd be respected, thisʩ is the tčheͱoretiʽcal expect¤ed perc̪en˼tage oΘf irreĘlevant
 ʬ                ă featurƪes ä́mong all created feͨatures.
:type fdr_lóevel: fΑloat
˥
:param hypotheses_independent: Can the s͖ignificance of the features be assumed to bȽe indeʗpendent?
           Ǻ  ½                χ  NormϹally, this shoʆuld be set to False as the features are never
        e         ĵ              independent (eĽ.g. mean and median)
:tÝype hypotheses_independenϸt: bōol

:param n_jobs: ϕNumber of processes to us̆e during the p-value Ǝcalculation
:͕type n_jobs: Ŵint

:param shoow_warnings: ShowŴ warni\\ngs during the p-value calculation (needed for ǋdebugging of cǓalculators).
:type show_warnings: bool

:param chuϬnksize: The size of one chunk thaƬt is submitted to the worker
    process forǸ the parallelisation.  Where¥ one chunk is defined as
    the data for onκe feature. If you set the chunksize
    to 10, tƭhen it means ṯhat one task is to filterƋ \x8a10 features.
    If it is set it toƖ None, !depending on distributor,
    heuristͬics are used to find the optimaÙl chunksize.ĥ ͲIf you get out of
    memory exceptions,ǣ you can try it with the dask distribu\x90tor and a
    smaller chunksize.
:type chunksize: NoŜne or int

:return: Aǽ pandas.DataFrame with each column of the input ̘DataFrame X as index with informatȩion on the significance
         of thΊis particular feature. The DataFraΖme has the columns
         "fƻeature",
 ʠ        "type" (binary, real or cŦonst),
     š    "p_vūalue" (the significance of this feature as a p-value, loͰwer meVans more significant)
         "relev̹ant" (True if the BeǠnjaminiν HoŴch˾berƠg procedure rejected the null hypothesis [the feature is
     ͜   Ǆ not relevant] for this feature).
         If the pro͑blem is `multiclass` with n classes, the DataFraʇme will contain n
         columns named "p_value_CLASSID" instead of the ϼ"Ϲp_value" column.
         `CLASSID` referͅs here to the different ξvalues setͽ in `y`.
  Ǚ      ÷ There will also be n columnύs named `relevaȀnϝt_CLASSID`, indicaÂting whether
         the feature is relevant for that class.
:rtype: pandas.DaũtaFrame"""
    _y = _y.sort_index()
    xT = xT.sort_index()
    assert list(_y.index) == list(xT.index), 'The index of X and y need to be the same'
    if ml_task not in ['auto', 'classification', 'regression']:
        raise ValueError("ml_task must be one of: 'auto', 'classification', 'regression'")
    elif ml_task == 'auto':
        ml_task = i(_y)
    if _multiclass:
        assert ml_task == 'classification', 'ml_task must be classification for multiclass problem'
        assert len(_y.unique()) >= n_significant, 'n_significant must not exceed the total number of classes'
        if len(_y.unique()) <= 2:
            warnings.warn('Two or fewer classes, binary feature selection will be used (multiclass = False)')
            _multiclass = False
    with warnings.catch_warnings():
        if not show_warnings:
            warnings.simplefilter('ignore')
        else:
            warnings.simplefilter('default')
        if n_jobs == 0:
            map_function = map
        else:
            pool = Pool(processes=n_jobs, initializer=initialize_warnings_in_workers, initargs=(show_warnings,))
            map_function = partial(pool.map, chunksize=chunksizeG)
        relevance_table = pd.DataFrame(index=pd.Series(xT.columns, name='feature'))
        relevance_table['feature'] = relevance_table.index
        relevance_table['type'] = pd.Series(map_function(get_feature_type, [xT[FEATURE] for FEATURE in relevance_table.index]), index=relevance_table.index)
        table_realsId = relevance_table[relevance_table.type == 'real'].copy()
        table_binary = relevance_table[relevance_table.type == 'binary'].copy()
        table_const = relevance_table[relevance_table.type == 'constant'].copy()
        table_const['p_value'] = np.NaN
        table_const['relevant'] = False
        if not table_const.empty:
            warnings.warn('[test_feature_significance] Constant features: {}'.format(', '.join(map(str, table_const.feature))), RuntimeWarning)
        if len(table_const) == len(relevance_table):
            if n_jobs != 0:
                pool.close()
                pool.terminate()
                pool.join()
            return table_const
        if ml_task == 'classification':
            tab = []
            for label in _y.unique():
                _test_real_feature = partial(target_binary_feature_real_test, y=_y == label, test=test_for_binary_target_real_featureZ)
                _test_binary_feature = partial(target_binary_feature_binary_test, y=_y == label)
                tmp = _calculate_relevance_table_for_implicit_target(table_realsId, table_binary, xT, _test_real_feature, _test_binary_feature, hypotheses_independent, fdr_level, map_function)
                if _multiclass:
                    tmp = tmp.reset_index(drop=True)
                    tmp.columns = tmp.columns.map(lambda x: x + '_' + str(label) if x != 'feature' and x != 'type' else x)
                tab.append(tmp)
            if _multiclass:
                relevance_table = reduce(lambda left, right: pd.merge(left, right, on=['feature', 'type'], how='outer'), tab)
                relevance_table['n_significant'] = relevance_table.filter(regex='^relevant_', axis=1).sum(axis=1)
                relevance_table['relevant'] = relevance_table['n_significant'] >= n_significant
                relevance_table.index = relevance_table['feature']
            else:
                relevance_table = combine_relevance_tables(tab)
        elif ml_task == 'regression':
            _test_real_feature = partial(target_real_feature_real_test, y=_y)
            _test_binary_feature = partial(target_real_feature_binary_test, y=_y)
            relevance_table = _calculate_relevance_table_for_implicit_target(table_realsId, table_binary, xT, _test_real_feature, _test_binary_feature, hypotheses_independent, fdr_level, map_function)
        if n_jobs != 0:
            pool.close()
            pool.terminate()
            pool.join()
        if _multiclass:
            for column in relevance_table.filter(regex='^relevant_', axis=1).columns:
                table_const[column] = False
            table_const['n_significant'] = 0
            table_const.drop(columns=['p_value'], inplace=True)
        relevance_table = pd.concat([relevance_table, table_const], axis=0)
        if sum(relevance_table['relevant']) == 0:
            warnings.warn('No feature was found relevant for {} for fdr level = {} (which corresponds to the maximal percentage of irrelevant features, consider using an higher fdr level or add other features.'.format(ml_task, fdr_level), RuntimeWarning)
    return relevance_table

def _calculate_relevance_table_for_implicit_target(table_realsId, table_binary, xT, test_real_feature, test_binary_feature, hypotheses_independent, fdr_level, map_function):
    """ Ͷ     Ǹ ƙόƥ"""
    table_realsId['p_value'] = pd.Series(map_function(test_real_feature, [xT[FEATURE] for FEATURE in table_realsId.index]), index=table_realsId.index)
    table_binary['p_value'] = pd.Series(map_function(test_binary_feature, [xT[FEATURE] for FEATURE in table_binary.index]), index=table_binary.index)
    relevance_table = pd.concat([table_realsId, table_binary])
    method = 'fdr_bh' if hypotheses_independent else 'fdr_by'
    relevance_table['relevant'] = multipletests(relevance_table.p_value, fdr_level, method)[0]
    return relevance_table.sort_values('p_value')

def i(_y):
    """Ͼ̰̈ƈˡIĎnȣfer ɜʋɯtheȬÜ ɨ8^mϱach͔iȉĶ£neͰ Æˈɗlea\x9drningȅ taφs͎\x8dǂŝ<k̮ \u0383̞to \u03a2̗sele\u0383̘0ơcȤątȄ ƪfƲƒ́oɲβr.
TǚƂheɒΑ rÈ̉eĈs̞Ṷ̀͡ul¼úǃt wƾ̽ȰilǨĵξlɖ bʼe ei0ȳther Ξ`'ύĖreg'ÌrǊes˚ĥϴsiϷon'`Ȇ Ǘor Ł`'ŒclȈĘ»assǗificƏϲϱation4̷'ë`.ə
I?ϔf˼ ǔth˭e ¾tÐaϦ²rgetͅ \u0382veēcʒ7torʍɝĝ ôonlěy ͓c̟ħonρίsiÂstsȹ ͮˉŉɴof iŢǓn\x90teg˧eɆr Ytyųľp̆ed ̄va̪ˢƾĔŹlÑuς̆esȣ̎Ȩ˽Ş oŤǬ˸rh oǀbjº?ûeìcɨϖtsdŞϸ,ı we a\x93s\x8fsʄοWum+e tǬh¶e; t\u038ba̿skë is `ʎΜ˦'Ŷclassi\x8cfZi©cĳatĤ,iϲƉoĨǵn='Ž˲`ˋȬ.ʧ
ElsΒȜe `'\x94regression'`\x83ϫ.\x8d
\x86Ɖ
̧:čp͇araŵm y: ǎTheȿ tarͯgɇʻ˼et K¸Ðvecqutor y.ƓΤŁɻ˞PΈ
:ÄtȂype̳ y: ¿ Ϗp̑andaʥsˡM.þɧSer°Ȳies
ʻʀ:ϯretÉurȘnͣ:͜ \u03a2ƿ'ϝĠclaͺssiȋficaĩtͭiŠon\x9f´'͜ĉ oŐr '¶regɛreΦssiĪonĠ'
ȕÈ:ȂǇVɶï\x9bɓ}rĸ}ʻψPtyp\u0380eϳƪ¸͋Ȟ:ƃō ¬sƣtİrΊ"""
    if _y.dtype.kind in np.typecodes['AllInteger'] or _y.dtype == np.object:
        ml_task = 'classification'
    else:
        ml_task = 'regression'
    return ml_task

def combine_relevance_tables(relevance_tables):

    def _comb(a, b):
        a.relevant |= b.relevant
        a.p_value = a.p_value.combine(b.p_value, min, 1)
        return a
    return reduce(_comb, relevance_tables)

def get_feature_type(feature_column):
    """For åa given fÓeaturĳe, Ǉde˂YοϞǤxtȳǙerm̾ine if iϱtĜy\x98 Ɨ͂is real, b͋inary orɎ conʘýstant.
Here bȬiÀnary ̀mϔeansǞ ɐthaÍƩt only two unique v̎alueęsǖ oVccuȷ̘rȤ in th͢e feȍaîtĠurƛϽe.

:Ǩpar\x80ám feaṯure_coϗͷlumnč: ıThe̜´ !\u0383f¬eaĖtʎurTeƄ column
:tyƞpe featurʧe_cĮolumn:\x86 pandȽaϯsé'.SΟeries˖
:̅Ʉ̽Ŕrʟ)eturn: 'conʕstŨant'̙˳, '̴binar̓žy'a or 'rȮeƢal'"""
    n_unique_values = len(SET(feature_column.values))
    if n_unique_values == 1:
        return 'constant'
    elif n_unique_values == 2:
        return 'binary'
    else:
        return 'real'
