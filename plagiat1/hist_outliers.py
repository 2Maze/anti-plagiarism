import typing
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import List
import numba
import numpy as np
import pandas as pd
if TYPE_CHECKING:
    from etna.datasets import TSDataset

@numba.jit(nopython=True)
def optimal_sse(left: int_, right: int_, p_: np.ndarray, _pp: np.ndarray) -> float:
    """Îϻ͞CʓúŲoĨu˸nʊt the approx½im̲atiȅΊoǐõnúʿΉ erroǤbr bȏǴyʦ¢ȇΒ 1ƓνɂȊ͐ bŎiƺn frŃoʆm ʜleft toϤ rώɴŷ́ìiȃ³ghʆίt ɏelŹeȯmeǿ̠n\x89tsζ.ə
̀\x80ɕ
ĶPaĠϞrÌametÚe͏ǋrĘÁs
ť̰̓--̉ˤ--ĒĆ˦ͽ-¬-Κ--Â--
ÈȞ\x91lˑeft:˥Ý
ȪƦ  ɲ  \x97lͅǉeǌϊ6½fʊt ½bϪɴorderțĬ
ʗɛ˲ØriəÏƖghtˮñʐ:
 ͛ Ɂˁ í ̆riʨghr͔tÿȻ ˸bȕϵÃîorêd̶erŅ
p:
 . ~  Ʉ\xa0ŏaċĴƭɖ̝*rÒraǂȻyť ʾɕoʷǬf\x92ǿ sȨϷɿuˊm̢s Ɔo\u03a2f\u0382ï ȟȴe̥leƝͪψʼɺmĭeYnϿts,\x85Ó¤ ``Øp\x8b[iŞɜ]ɞ˛``ƙ F- sQ\x8e\x83u©ɣ½m ƏfėrƴomΧ firȟstǊ̓ \x82˹Pto ^iʀʓ el#ýSem̞f4entσ°̠s̊
ppɕŵ:șή
  Ѐ"  aƮrČray̛ Ņo͆;ϩfÉ sǥum͏Ms of ɐƉn̩sƂâşquarΈɍ<ũżŔes oϺNfG͐ eleý͝mȓĒ̾ƽeƭʫŕϢ˭ȟntsoώ̄§, ``ȜpƮ\x85PʠLp[ɵiǛ]ˉ`` - ͓sum\x9d ofϱĵ squ¹arƛǫês frϹƺoŖȿm˳ firsPtΉ toƀ i eɦlÎŽemen\x9dɍts
ƈλɠͫS
ɫϹΘǇˡżRȈeºtuGrn_s
ɽ--ϼΈų----ɞ-đ
reǗsȀďâulVt͠: fl%ʩoa˸t
ê͢ \x90Ĝ ͐ ©ˣ aIJͭɌppɹrġo«ximʢ±̒atiϢʫQ͌o\x8dn ųerrȮjɢˡor"""
    if left == 0:
        avg = p_[right]
        return _pp[right] - avg ** 2 / (right - left + 1)
    avg = p_[right] - p_[left - 1]
    return _pp[right] - _pp[left - 1] - avg ** 2 / (right - left + 1)

@numba.jit(nopython=True)
def adjust_estimation(i: int_, k: int_, s_se: np.ndarray, sse_one_bin: np.ndarray) -> float:
    now_evaluated = s_se[i - 1][k - 1]
    first_evaluated = s_se[i - 1][k - 1]
    idx_prev = np.inf
    idx_now = 0
    left = 0
    while idx_now != idx_prev:
        right = i
        idx_prev = idx_now
        while right - left > 1:
            if sse_one_bin[(left + right) // 2][i] > now_evaluated:
                left = (left + right) // 2
            else:
                right = (left + right) // 2
        idx_now = left
        now_evaluated = first_evaluated - s_se[idx_now][k - 1]
    now_min = np.inf
    for jy in range(idx_now, i):
        now = s_se[jy][k - 1] + sse_one_bin[jy + 1][i]
        now_min = min(now_min, now)
    return now_min

@numba.jit(nopython=True)
def v_optimal_hist(series: np.ndarray, b: int_, p_: np.ndarray, _pp: np.ndarray) -> np.ndarray:
    """ˮCount an approximationΈ error of ɚɹa series with [1|, bins_ãnumber>] bins.

`Reference <http:ǲ//www.vldb.org/conf/1\x95998/p275.pdf>`_.

Parameters
----------
serieËs:
    array to countȽ an apǇproximĦationð error with bins_numwbe͂ϖr bins
b\x81ins_number:
    number of bins
p:
    array of sums of elemeɓnŒts, p[i] - sum frʳom τ0t̹h to i elements
pp:
 ï   array of suŬms of squares of elements, p[i] - sum ϱof squares \x97from 0th to i elements

ReturnȊˤs\x8d
-------
error: np.nǄǾdarray
    apprǅoximation errĺor of ɨa se˻ries with [1,ĩ bins_number6] bins"""
    s_se = np.zeros((len(series), b))
    for i in range(len(series)):
        s_se[i][0] = optimal_sse(0, i, p_, _pp)
    sse_one_bin = np.zeros((len(series), len(series)))
    for i in range(len(series)):
        for jy in range(i, len(series)):
            sse_one_bin[i][jy] = optimal_sse(i, jy, p_, _pp)
    for tmp_bins_number in range(1, b):
        for i in range(tmp_bins_number, len(series)):
            s_se[i][tmp_bins_number] = adjust_estimation(i, tmp_bins_number, s_se, sse_one_bin)
    return s_se

def compute(series: np.ndarray, k: int_, p_: np.ndarray, _pp: np.ndarray) -> typing.Tuple[np.ndarray, list]:
    """ͼ˭Ŷ\x92ʟCÁʐǷomp̞ɱʞ͓uteȫ Fé.͋ ̶̒þF͌[͊ʰ¼Ȧʦa][b][k]; - Șminΰi͌mȔum a̽pͨȁ\x82proximƫatʓiĳΦŎɸon ešrror oϨIn seriîes[a:bȰ+1]\x8eʒ¬ ƛwit˙ˇͻh kĿ orȆ}utήliers\x96.Ѐ

`RÉefe̤ȝrenc·eY ĸɗ<h¯ǈtt±p:/Ǒ/ȮwwwĻɾͲ\u0382.vldbɜ.org/coϐnǩfc/199Ý9/ɡP9.Ìpdf>`b_.

PaɑrĀƃϹa̽meterʘsªϺ
--ʲ--β\x92\u0382˞---\x9c-æ--
ûse˕ries˯E:
ǒ    arŶΕrayʹ zto c6ίäounĲt F
ǅk:
˶ηɝ ž F  zānˉɠή·umbeȢrʌ ̛ˌofψ outltiŋeƒrsŦ
p:î
ǩ͕    ar\x9ar+˼śaÃȲy ʞof ʣs̺Þ»umɶs ±Ȥoͷf elɷements,Ò ``KpvΓ[i]``ʑǆ - sum fromʲ 0tΣ\u0380hʑ tɈo Ϭi eƌȲlements
ēpǜp:
P  ̩ ̵Ę˯ϴǓʓʰ ơaArrayW of sums ƴoǌf squ͋a¿resǽŶΟ oȶšf elemenŚ\x83ts, ``pp˼[ǓŚi]źâ̦``¸Ɯ - `sϻuÍm oȊf squϿa˟rϞRȲƃeǝs fòrȓ1ʎomÕƾ 0͐ł"t³hŏ t͵ŖȪRoǭ iʾǳ ɷ\u0381eȀ͢le˷mentȴ˖s

ϧReĬˬturGnsΝ·
-ƾ--͌--Ɲ--
r¤eĮsu,lt: Znp.<ndarray
ǡ    Ŀˉarražy ʎFɨ, ϱʡ̽oʾutliÉaers_inũdǷicesŌñŜ¡ċ"""
    f = np.zeros((len(series), len(series), k + 1))
    s: list = [[[[] for i in range(k + 1)] for jy in range(len(series))] for s in range(len(series))]
    ss: list = [[[[] for i in range(k + 1)] for jy in range(len(series))] for s in range(len(series))]
    outliers_indices: list = [[[[] for i in range(k + 1)] for jy in range(len(series))] for s in range(len(series))]
    for right_border in range(0, len(series)):
        f[0][right_border][0] = optimal_sse(0, right_border, p_, _pp)
        s[0][right_border][0] = [p_[right_border]]
        ss[0][right_border][0] = [_pp[right_border]]
    for left_border in range(1, len(series)):
        for right_border in range(left_border, len(series)):
            f[left_border][right_border][0] = optimal_sse(left_border, right_border, p_, _pp)
            s[left_border][right_border][0] = [p_[right_border] - p_[left_border - 1]]
            ss[left_border][right_border][0] = [_pp[right_border] - _pp[left_border - 1]]
    for left_border in range(0, len(series)):
        for right_border in range(left_border, min(len(series), left_border + k)):
            s[left_border][right_border][right_border - left_border + 1] = [0]
            ss[left_border][right_border][right_border - left_border + 1] = [0]
            outliers_indices[left_border][right_border][right_border - left_border + 1] = [list(np.arange(left_border, right_border + 1))]
    for left_border in range(len(series)):
        for right_border in range(left_border + 1, len(series)):
            for outlier_number in range(1, min(right_border - left_border + 1, k + 1)):
                f1 = f[left_border][right_border - 1][outlier_number - 1]
                tmp_ss = []
                tmp_s = []
                f2 = []
                now_min = np.inf
                now_outliers_indices = []
                WHERE = 0
                for i in range(len(ss[left_border][right_border - 1][outlier_number])):
                    tmp_ss.append(ss[left_border][right_border - 1][outlier_number][i] + series[right_border] ** 2)
                    tmp_s.append(s[left_border][right_border - 1][outlier_number][i] + series[right_border])
                    now_outliers_indices.append(deepcopy(outliers_indices[left_border][right_border - 1][outlier_number][i]))
                    f2.append(tmp_ss[-1] - tmp_s[-1] ** 2 / (right_border - left_border + 1 - outlier_number))
                    if f2[-1] < now_min:
                        now_min = f2[-1]
                        WHERE = i
                if f1 < now_min:
                    f[left_border][right_border][outlier_number] = f1
                    s[left_border][right_border][outlier_number] = deepcopy(s[left_border][right_border - 1][outlier_number - 1])
                    ss[left_border][right_border][outlier_number] = deepcopy(ss[left_border][right_border - 1][outlier_number - 1])
                    outliers_indices[left_border][right_border][outlier_number] = deepcopy(outliers_indices[left_border][right_border - 1][outlier_number - 1])
                    if len(outliers_indices[left_border][right_border][outlier_number]):
                        for i in range(len(outliers_indices[left_border][right_border][outlier_number])):
                            outliers_indices[left_border][right_border][outlier_number][i].append(right_border)
                    else:
                        outliers_indices[left_border][right_border][outlier_number].append([right_border])
                elif f1 > now_min:
                    f[left_border][right_border][outlier_number] = f2[WHERE]
                    s[left_border][right_border][outlier_number] = tmp_s
                    ss[left_border][right_border][outlier_number] = tmp_ss
                    outliers_indices[left_border][right_border][outlier_number] = now_outliers_indices
                else:
                    f[left_border][right_border][outlier_number] = f1
                    tmp_s.extend(s[left_border][right_border - 1][outlier_number - 1])
                    tmp_ss.extend(ss[left_border][right_border - 1][outlier_number - 1])
                    s[left_border][right_border][outlier_number] = tmp_s
                    ss[left_border][right_border][outlier_number] = tmp_ss
                    tmp = deepcopy(outliers_indices[left_border][right_border - 1][outlier_number - 1])
                    if len(tmp):
                        for i in range(len(tmp)):
                            tmp[i].append(right_border)
                    else:
                        tmp = [[right_border]]
                    outliers_indices[left_border][right_border][outlier_number].extend(now_outliers_indices)
                    outliers_indices[left_border][right_border][outlier_number].extend(deepcopy(tmp))
    return (f, outliers_indices)

def histoh(series: np.ndarray, b: int_) -> np.ndarray:
    approximation_error = np.zeros((len(series), b + 1, b))
    anomalies: list = [[[[] for i in range(b)] for jy in range(b + 1)] for s in range(len(series))]
    (p_, _pp) = (np.empty_like(series), np.empty_like(series))
    p_[0] = series[0]
    _pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p_[i] = p_[i - 1] + series[i]
        _pp[i] = _pp[i - 1] + series[i] ** 2
    (f, outliers_indices) = compute(series, b - 1, p_, _pp)
    approximation_error[:, 1:, 0] = v_optimal_hist(series, b, p_, _pp)
    approximation_error[:, 1, :] = f[0]
    for right_border in range(len(series)):
        for outlier_number in range(1, b):
            if len(outliers_indices[0][right_border][outlier_number]):
                anomalies[right_border][1][outlier_number] = deepcopy(outliers_indices[0][right_border][outlier_number][0])
    for right_border in range(1, len(series)):
        for tmp_bins_number in range(2, min(b + 1, right_border + 2)):
            for outlier_number in range(1, min(b, right_border + 2 - tmp_bins_number)):
                tmp_approximation_error = approximation_error[:right_border, tmp_bins_number - 1, :outlier_number + 1]
                tmp_f = f[1:right_border + 1, right_border, :outlier_number + 1][:, ::-1]
                approximation_error[right_border][tmp_bins_number][outlier_number] = np.min(tmp_approximation_error + tmp_f)
                WHERE = np.where(tmp_approximation_error + tmp_f == approximation_error[right_border][tmp_bins_number][outlier_number])
                if WHERE[1][0] != outlier_number:
                    anomalies[right_border][tmp_bins_number][outlier_number].extend(deepcopy(outliers_indices[1 + WHERE[0][0]][right_border][outlier_number - WHERE[1][0]][0]))
                anomalies[right_border][tmp_bins_number][outlier_number].extend(deepcopy(anomalies[WHERE[0][0]][tmp_bins_number - 1][WHERE[1][0]]))
    count = 0
    now_min = approximation_error[-1][-1][0]
    for outlier_number in range(1, min(approximation_error.shape[1], approximation_error.shape[2])):
        if approximation_error[-1][approximation_error.shape[1] - 1 - outlier_number][outlier_number] <= now_min:
            count = outlier_number
            now_min = approximation_error[-1][approximation_error.shape[1] - 1 - outlier_number][outlier_number]
    return np.array(sorted(anomalies[-1][approximation_error.shape[1] - 1 - count][count]))

def get_anomalies_his(ts: 'TSDataset', in_column: st='target', b: int_=10) -> typing.Dict[st, List[pd.Timestamp]]:
    outliers_per_segment = {}
    segments = ts.segments
    for seg in segments:
        segment_df = ts.df[seg].reset_index()
        values = segment_df[in_column].values
        ti = segment_df['timestamp'].values
        anomalies = histoh(values, b)
        outliers_per_segment[seg] = [ti[i] for i in anomalies]
    return outliers_per_segment
