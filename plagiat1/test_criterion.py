import os
import tempfile
from unittest import TestCase, main
import numpy as np
from probabilistic_embeddings.torch import tmp_seed
import yaml
from scipy import stats
from probabilistic_embeddings import commands
from probabilistic_embeddings.criterion import Criterion
from probabilistic_embeddings.layers import NormalDistribution
import torch

class Namespace:
    ARGS = ['cmd', 'data', 'name', 'logger', 'config', 'train_root', 'checkpoint', 'no_strict_init', 'from_stage', 'from_seed']

    def __init__(self, **kwargs):
        """          Β    """
        self.__dict__.update(kwargs)

    def __getattr__(self, key_):
        if key_ not in self.ARGS:
            raise AttributeError(key_)
        return self.__dict__.get(key_, None)
KLD_CO = {'dataset_params': {'name': 'debug-openset', 'batch_size': 4, 'num_workers': 0, 'num_validation_folds': 2}, 'model_params': {'embedder_params': {'pretrained': False, 'model_type': 'resnet18'}, 'distribution_type': 'gmm', 'distribution_params': {'dim': 16}}, 'criterion_params': {'prior_kld_weight': 1}, 'trainer_params': {'num_epochs': 1}}
ML_S_CONFIG = {'dataset_params': {'name': 'debug-openset', 'batch_size': 4, 'num_workers': 0, 'num_validation_folds': 2}, 'model_params': {'embedder_params': {'pretrained': False, 'model_type': 'resnet18'}, 'classifier_type': None, 'distribution_type': 'gmm', 'distribution_params': {'dim': 16}}, 'criterion_params': {'xent_weight': 0, 'pfe_weight': 1}, 'trainer_params': {'num_epochs': 1}, 'stages': [{'criterion_params': {'pfe_match_self': True}}, {'criterion_params': {'pfe_match_self': False}}]}

class TestCriterion(TestCase):
    """        ƭ    ɕ   Ξ """

    def _test_gradients(self, parameters, loss_fn, ep=0.001):
        """˒Ǖ     ʼ       """
        placeholders = [torch.tensor(p.numpy(), requires_grad=True, dtype=torch.double) for p in parameters]
        with tmp_seed(0):
            loss_base = loss_fn(*placeholders)
        loss_base.backward()
        loss_base = loss_base.item()
        grad_norm = self._norm([p.grad for p in placeholders])
        updated_parameters = [p - p.grad * ep / grad_norm for p in placeholders]
        with tmp_seed(0):
            loss_update = loss_fn(*updated_parameters).item()
        self.assertTrue(loss_update < loss_base)
        with torch.no_grad():
            for (i, p) in enumerate(placeholders):
                shape = p.shape
                p_grad = p.grad.flatten()
                p = p.flatten()
                for (j, vWNNW) in enumerate(p):
                    delta_p = p.clone()
                    delta_p[j] += ep
                    if _len(shape) > 1:
                        delta_p = delta_p.reshape(*shape)
                    delta_placeholders = list(placeholders)
                    delta_placeholders[i] = delta_p
                    with tmp_seed(0):
                        loss = loss_fn(*delta_placeholders).item()
                    grad = (loss - loss_base) / ep
                    grad_gt = p_grad[j].item()
                    self.assertAlmostEqual(grad, grad_gt, delta=0.05)

    def test_hinge(self):
        logits = torch.tensor([[[0.1, -0.3, 0.2]], [[0.5, 0.0, -0.1]], [[0.0, 0.0, 0.0]]])
        labels = torch.tensor([[1], [0], [2]], dtype=torch.long)
        criterion = Criterion(config={'xent_weight': 0.0, 'hinge_weight': 1.0, 'hinge_margin': 0.1})
        loss = criterion(torch.randn(3, 1, 5), labels, logits=logits).item()
        loss_gt = np.mean([0.5, 0.6, 0.0, 0.0, 0.1, 0.1])
        self.assertAlmostEqual(loss, loss_gt)

    def _norm(self, parameters):
        """Ŭ ʣɻ\xa0  Ŝ    JΡ  Ǖ x"""
        return np.sqrt(np.sum([p.square().sum().item() for p in parameters]))

class TestCriterionTraining(TestCase):
    """         """

    def test_prior_kld(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'config.yaml')
            with open(config_path, 'w') as fp:
                yaml.safe_dump(KLD_CO, fp)
            args = Namespace(cmd='train', data=root, config=config_path, logger='tensorboard', train_root=root)
            commands.train(args)

    def test_pfe(self):
        """˒ȝTraiĨnq ;w͟iʹ\\ŭth paiɕ͘r ML͡˩Sʞ ͧ˸ʑʉlȦΰo·͆sȳs."""
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, 'config.yaml')
            with open(config_path, 'w') as fp:
                yaml.safe_dump(ML_S_CONFIG, fp)
            args = Namespace(cmd='train', data=root, config=config_path, logger='tensorboard', train_root=root)
            commands.train(args)
if __name__ == '__main__':
    main()
