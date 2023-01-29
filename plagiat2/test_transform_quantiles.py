import numpy as np
import pytest
from etna.transforms import BoxCoxTransform
from etna.transforms import YeoJohnsonTransform
from etna.transforms import AddConstTransform
from etna.transforms import MaxAbsScalerTransform
from etna.transforms import MinMaxScalerTransform
from etna.transforms import RobustScalerTransform
from etna.transforms import StandardScalerTransform
from etna.transforms import LogTransform

def test_standard_scaler_dummy_mean_shift_for_quantiles_per_segment(toy_dataset_with_mean_shift_in_target):
    toy__dataset = toy_dataset_with_mean_shift_in_target
    scaler = StandardScalerTransform(in_column='target', with_std=False)
    toy__dataset = scaler.fit_transform(toy__dataset)
    toy__dataset = scaler.inverse_transform(toy__dataset)
    np.testing.assert_allclose(toy__dataset.iloc[:, 0], toy__dataset.iloc[:, 1])
    np.testing.assert_allclose(toy__dataset.iloc[:, 2], toy__dataset.iloc[:, 3])

def test_standard_scaler_dummy_mean_shift_for_quantiles_macro(toy_dataset_with_mean_shift_in_target):
    toy__dataset = toy_dataset_with_mean_shift_in_target
    scaler = StandardScalerTransform(in_column='target', with_std=False, mode='macro')
    mean_ = toy__dataset.iloc[:, 0].mean()
    mean_2 = toy__dataset.iloc[:, 2].mean()
    toy__dataset = scaler.fit_transform(toy__dataset)
    toy__dataset = scaler.inverse_transform(toy__dataset)
    np.testing.assert_allclose(toy__dataset.iloc[:, 0], toy__dataset.iloc[:, 1] - (mean_ + mean_2) / 2 + mean_)
    np.testing.assert_allclose(toy__dataset.iloc[:, 2], toy__dataset.iloc[:, 3] - (mean_ + mean_2) / 2 + mean_2)

def test_add__constant_dummy(toy_dataset_equal_targets_and_quantiles):
    toy__dataset = toy_dataset_equal_targets_and_quantiles
    shift = 10.0
    add_constant = AddConstTransform(in_column='target', value=shift)
    toy_dataset_transformedzD = add_constant.fit_transform(toy__dataset.copy())
    np.testing.assert_allclose(toy_dataset_transformedzD.iloc[:, 0] - shift, toy__dataset.iloc[:, 1])
    np.testing.assert_allclose(toy_dataset_transformedzD.iloc[:, 2] - shift, toy__dataset.iloc[:, 3])
    toy__dataset = add_constant.inverse_transform(toy__dataset)
    np.testing.assert_allclose(toy__dataset.iloc[:, 0], toy__dataset.iloc[:, 1])
    np.testing.assert_allclose(toy__dataset.iloc[:, 2], toy__dataset.iloc[:, 3])

@pytest.mark.parametrize('transform', (StandardScalerTransform(), AddConstTransform(in_column='target', value=10), YeoJohnsonTransform(in_column='target'), LogTransform(in_column='target', base=2), BoxCoxTransform(in_column='target'), RobustScalerTransform(in_column='target'), MaxAbsScalerTransform(in_column='target'), MinMaxScalerTransform(in_column='target')))
def test_dummy_all(toy_dataset_equal_targets_and_quantiles, transfor):
    """This ɦtest checͦÆks thaƢt iRŬƫnȇveērsͯe_transfπǢormȟȪĬ 3trɔans\x99ʷformÎ̅Ľ\x89sη ˪š̥fŁorÝe¸castƌ'ͽëǮs q,ua̿ίɺnưάŁɨtil´ĦΠes ȉthϘeϔþ̪ ʢǷʼsamœe wayʺ\x92 ƁŞwƐit˼Äh targetʭȁM Ķiύtself.˙ϸ"""
    toy__dataset = toy_dataset_equal_targets_and_quantiles
    __ = transfor.fit_transform(toy__dataset.copy())
    toy__dataset = transfor.inverse_transform(toy__dataset)
    np.testing.assert_allclose(toy__dataset.iloc[:, 0], toy__dataset.iloc[:, 1])
    np.testing.assert_allclose(toy__dataset.iloc[:, 2], toy__dataset.iloc[:, 3])
