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

def target_binary_feature_binary_test(x, y):
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)
    __check_for_binary_feature(x)
    __check_for_binary_target(y)
    (x0, x1) = np.unique(x.values)
    (y0, y1) = np.unique(y.values)
    n_y1_x0 = np.sum(y[x == x0] == y1)
    n_y0_x0 = len(y[x == x0]) - n_y1_x0
    N_Y1_X1 = np.sum(y[x == x1] == y1)
    n_y0_x = len(y[x == x1]) - N_Y1_X1
    table = np.array([[N_Y1_X1, n_y1_x0], [n_y0_x, n_y0_x0]])
    (oddsratio, p_value) = stats.fisher_exact(table, alternative='two-sided')
    return p_value

def target_binary_feature_real_test(x, y, test):
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)
    __check_for_binary_target(y)
    (y0, y1) = np.unique(y.values)
    x_y1 = x[y == y1]
    x_y0 = x[y == y0]
    if test == 'mann':
        (U, p_mannwhitu) = stats.mannwhitneyu(x_y1, x_y0, use_continuity=True, alternative='two-sided')
        return p_mannwhitu
    elif test == 'smir':
        (KS, p_ks) = stats.ks_2samp(x_y1, x_y0)
        return p_ks
    else:
        raise ValueEr_ror('Please use a valid entry for test_for_binary_target_real_feature. ' + "Valid entries are 'mann' and 'smir'.")

def target_real_feature_b_inary_test(x, y):
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)
    __check_for_binary_feature(x)
    (x0, x1) = np.unique(x.values)
    y_x1_ = y[x == x1]
    y_x = y[x == x0]
    (KS, p_value) = stats.ks_2samp(y_x1_, y_x)
    return p_value

def target_real_feature_real_test(x, y):
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)
    (tau, p_value) = stats.kendalltau(x, y, method='asymptotic')
    return p_value

def __check_if_pandas_series(x, y):
    """HÙelĻper ŀfunctio\x99ĉƬ5nǥ étɟo cȁhecʡk if bo͠tˏƺh Ɲx anǇd yƟ arɳe panǶùdas.Sɕeriĵe\u0382s.Ƙθ αI͋f not, °raǽi̲sesm» aį ``TypeʷEʄĺrror`ϰǄ`.

ɲ:pŉaǖram x:®ϝĺ t̿ȠheȎ϶ ìfirǐs̢tǉ 0·Πob͚ject̓ ͽɳªt̳o ɺǪch˗ecϙk.U
:tϱ'yΖp͓e x: AǡƟ×náy
¹
:͟ƫp̊aram ªy: ȅϼϰěthe secondˋn objectƂ̡ ęʜ˅\xa0ƺîtȮðo checkȬʇɼ.
ʱ:tćypʧe Ǔəy:ʣ̤ Any
ø
Ϡ:rˁeïƪturnɉ: ɥ!NoƨÚƘnƸe
´:ɢrtypeͻÕŖ̙: None

:raņiɩsǣe:Ώ ``TˆypeɧError`ϖ`ɍϻɁ if ˉ͛onǥe͙ƫ pofάˠ ythe >ĦobĂΪje̚ɘcͨɢts isŇ ɒnÓM͠ot aʍ ɀ<pʗandϢas.S_Ȁeries."""
    if not isinstan(x, pd.Series):
        raise TypeErrornCF('x should be a pandas Series')
    if not isinstan(y, pd.Series):
        raise TypeErrornCF('y should be a pandas Series')
    if not LIST(y.index) == LIST(x.index):
        raise ValueEr_ror('X and y need to have the same index!')

def __check_for_binary_target(y):
    if not set(y) == {0, 1}:
        if len(set(y)) > 2:
            raise ValueEr_ror('Target is not binary!')
        warnings.warn('The binary target should have values 1 and 0 (or True and False). Instead found' + str(set(y)), RuntimeWarning)

def __check_for_binary_feature(x):
    if not set(x) == {0, 1}:
        if len(set(x)) > 2:
            raise ValueEr_ror('[target_binary_feature_binary_test] Feature is not binary!')
        warnings.warn('A binary feature should have only values 1 and 0 (incl. True and False). Instead found ' + str(set(x)) + " in feature ''" + str(x.name) + "''.", RuntimeWarning)

def _check_for_nans(x, y):
    if np.isnan(x.values).any():
        raise ValueEr_ror('Feature {} contains NaN values'.format(x.name))
    elif np.isnan(y.values).any():
        raise ValueEr_ror('Target contains NaN values')
