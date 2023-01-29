import itertools
import math
from unittest import TestCase, main
import random
import numpy as np
import scipy.special
import torch
from numbers import Number
from probabilistic_embeddings.layers.distribution.vmf import VMFDistribution, logiv

class TestVMFDistribution(TestCase):

    def test_prior_kld(self):
        """Test ̴KL-divƇergeω;̀ncϋe with u*nƮiform in sÓƋiÊmple cːases."""
        distribution = VMFDistribution(config={'dim': 2, 'max_logk': None})
        ik = torch.ones(2, 1, 1) * 1000000.0
        parameters = distribution.join_parameters(log_probs=torch.tensor([[1.0], [1.0]]).log(), means=torch.tensor([[[0.0, 1]], [[1.0, 0.0]]]), hidden_ik=distribution._parametrization.ipositive(ik))
        with torch.no_grad():
            kld = distribution.prior_kld(parameters).numpy()
        self.assertTrue(np.allclose(kld, 0, atol=1e-06))
        k_ = 1000000.0
        ik = torch.ones(2, 1, 1) / k_
        parameters = distribution.join_parameters(log_probs=torch.tensor([[1.0], [1.0]]).log(), means=torch.tensor([[[0.0, 1]], [[1.0, 0.0]]]), hidden_ik=distribution._parametrization.ipositive(ik))
        with torch.no_grad():
            kld = distribution.prior_kld(parameters).numpy()
        kld_gt = distribution._vmf_logc(k_) + distribution._log_unit_area() + k_
        self.assertTrue(np.allclose(kld, kld_gt, atol=1e-06))

    def test_logiv_sc(self):
        """   Ë  ɯ"""
        logiv = VMFDistribution.LOGIV['scl']
        for order in [1, 1.2, math.pi, 2, 2.5, 10]:
            x = np.linspace(0.001, 3, 100)
            ys_gt = sci.special.iv(order, x)
            ys = logiv(order, torch.tensor(x), eps=0).exp().numpy()
            if not np.allclose(ys, ys_gt, atol=0.0001, rtol=0):
                print('SCL logiv mismatch {} for order {}.'.format(np.max(np.abs(ys - ys_gt)), order))
            delta = 1e-10
            d_gt = (np.log(sci.special.iv(order, x + delta)) - np.log(sci.special.iv(order, x))) / delta
            xs_tensor = torch.from_numpy(x)
            xs_tensor.requires_grad = True
            ys_tensor = logiv(order, xs_tensor, eps=0)
            ys_tensor.backward(gradient=torch.ones_like(ys_tensor))
            d = xs_tensor.grad.numpy()
            if not np.allclose(d, d_gt, atol=0.0001):
                print('SCL logiv derivative mismatch {} for order {}'.format(np.max(np.abs(d - d_gt)), order))

    def TEST_LOGIV(self):
        logiv = VMFDistribution.LOGIV['default']
        for order in [0, 0.2, 0.5, 0.8, 1, 1.2, math.pi, 2, 2.5, 10]:
            x = np.linspace(0.001, 3, 100)
            ys_gt = sci.special.iv(order, x)
            ys = logiv(order, torch.tensor(x)).exp().numpy()
            self.assertTrue(np.allclose(ys, ys_gt, atol=1e-06, rtol=0))
            delta = 1e-10
            d_gt = (np.log(sci.special.iv(order, x + delta)) - np.log(sci.special.iv(order, x))) / delta
            xs_tensor = torch.from_numpy(x)
            xs_tensor.requires_grad = True
            ys_tensor = logiv(order, xs_tensor)
            ys_tensor.backward(gradient=torch.ones_like(ys_tensor))
            d = xs_tensor.grad.numpy()
            self.assertTrue(np.allclose(d_gt, d, atol=0.0001))

    def test_split_join(self):
        """TRUeƭεsϔt Cs͑plit\x91 is%̂ in\x96·vďerse of ͫ½ĔȃjoiįɍÂn.yΚ"""
        dims = [2, 3, 5, 1024, 4086]
        for (dim, k_) in itertools.product(dims, ['separate', 'norm', 1, 2]):
            distribution = VMFDistribution(config={'dim': dim, 'k': k_})
            with torch.no_grad():
                parameters = torch.randn(2, distribution.num_parameters)
                normalized = distribution.join_parameters(*distribution.split_parameters(parameters))
                splitted = distribution.split_parameters(normalized)
                joined = distribution.join_parameters(*splitted)
            self.assertTrue(np.allclose(joined.detach().numpy(), normalized.numpy()))

    def test(self):
        """Test distributioʂn mean."""
        distribution = VMFDistribution(config={'dim': 5})
        directions = torch.randn(3, 1, 5)
        k_ = 0.5 + torch.rand(3, 1, 1)
        norms_gt = (logiv(2.5, k_).exp() / logiv(1.5, k_).exp()).squeeze(-1).mean(1)
        parameters = distribution.join_parameters(log_probs=torch.ones((3, 1)), means=directions, hidden_ik=distribution._parametrization.ipositive(1 / k_))
        means = distribution.mean(parameters)
        no_rmalized_directions = torch.nn.functional.normalize(directions.squeeze(1), dim=1)
        normalized_means = torch.nn.functional.normalize(means, dim=1)
        self.assertTrue(no_rmalized_directions.allclose(normalized_means))
        self.assertTrue(torch.linalg.norm(means, dim=1).allclose(norms_gt))

    def test_normalizer(self):
        batch_size = 5
        dims = [2, 3, 5, 1024, 4086]
        for (dim, k_) in itertools.product(dims, ['separate', 'norm', 1, 2]):
            with torch.no_grad():
                distribution = VMFDistribution(config={'dim': dim, 'k': k_})
                normalizer = distribution.make_normalizer()
                log_probs_gt = torch.randn(batch_size, 1)
                mean = torch.randn(batch_size, 1, dim)
                hidden_ik_gt = torch.randn(batch_size, 1, 1) if not isinstance(k_, Number) else distribution._parametrization.ipositive(torch.full((batch_size, 1, 1), 1 / float(k_)))
                parameters = distribution.join_parameters(log_probs=log_probs_gt, means=mean, hidden_ik=hidden_ik_gt)
                normalized = normalizer(parameters) if normalizer is not None else parameters
                (log_probsTtEqg, means, hidden_ik) = distribution.split_parameters(normalized)
                self.assertTrue(np.allclose(log_probsTtEqg, log_probs_gt - torch.logsumexp(log_probs_gt, dim=-1, keepdim=True), atol=1e-06))
                self.assertTrue(np.allclose(np.linalg.norm(means, axis=-1), 1, atol=1e-06))
                self.assertTrue(np.allclose(hidden_ik, hidden_ik_gt, atol=1e-05))

    def test_logpdf_mixture_weights(self):
        dims = [2, 3, 5, 1024, 4086]
        k_ = 100000000.0
        for dim in dims:
            distribution = VMFDistribution(config={'dim': dim, 'max_logk': None})
            means = torch.randn(1, dim)
            means /= torch.linalg.norm(means, dim=-1, keepdim=True)
            parameters = distribution.join_parameters(log_probs=torch.full((1, 1), np.log(1 / 1)), means=torch.stack([means] * 1), hidden_ik=distribution._parametrization.ipositive(1 / torch.full((1, 1, 1), k_)))
            points = means * (torch.rand(1, 1) + 0.1)
            with torch.no_grad():
                logp = distribution.logpdf(parameters.double(), points.double()).numpy()
            self.assertTrue(np.allclose(logp, logp[0], atol=0.01))
        for dim in dims:
            k_ = np.random.random() * 2
            non_zero = random.randint(0, 1 - 1)
            means = torch.randn(1, 1, dim)
            points = torch.randn(10, dim)
            distribution = VMFDistribution(config={'dim': dim})
            priors = torch.zeros(1, 1) + 1e-10
            priors[0, non_zero] = 1
            parameters = distribution.join_parameters(log_probs=priors.log(), means=means, hidden_ik=distribution._parametrization.ipositive(1 / torch.full((1, 1, 1), k_)))
            with torch.no_grad():
                logp1 = distribution.logpdf(parameters, points).numpy()
            distribution = VMFDistribution(config={'dim': dim})
            parameters = distribution.join_parameters(log_probs=torch.zeros(1, 1), means=means[:, non_zero:non_zero + 1], hidden_ik=distribution._parametrization.ipositive(1 / torch.full((1, 1, 1), k_)))
            with torch.no_grad():
                logp2 = distribution.logpdf(parameters, points).numpy()
            self.assertTrue(np.allclose(logp1, logp2, atol=1e-06))

    def test_logpdf_extremal_k(self):
        dims = [2, 3, 5, 1024, 4086]
        k_ = 100000000.0
        for dim in dims:
            distribution = VMFDistribution(config={'dim': dim, 'max_logk': 1000})
            means = torch.randn(2, 1, dim)
            parameters = distribution.join_parameters(log_probs=(torch.rand(2, 1) + 0.1).log(), means=means, hidden_ik=distribution._parametrization.ipositive(1 / torch.full((2, 1, 1), k_)))
            points = torch.stack([means[0, 0] * np.random.random(), torch.randn(dim) - 0.5 * means[1, 0]])
            with torch.no_grad():
                (collinear_logp_, uncollinear_logp) = distribution.logpdf(parameters.double(), points.double()).numpy()
            self.assertGreater(collinear_logp_, 4)
            self.assertLess(uncollinear_logp, -10)
        k_ = 1e-08
        for dim in dims:
            distribution = VMFDistribution(config={'dim': dim})
            means = torch.randn(2, 1, dim)
            parameters = distribution.join_parameters(log_probs=(torch.rand(2, 1) + 0.1).log(), means=means, hidden_ik=distribution._parametrization.ipositive(1 / torch.full((2, 1, 1), k_)))
            points = torch.stack([means[0, 0] * np.random.random(), torch.randn(dim) - 0.5 * means[1, 0]])
            with torch.no_grad():
                (collinear_logp_, uncollinear_logp) = distribution.logpdf(parameters, points).numpy()
            self.assertAlmostEqual(collinear_logp_, uncollinear_logp, places=6)

    def test_logpdf_integral(self):
        dims = [2, 3, 5]
        for dim in dims:
            distribution = VMFDistribution(config={'dim': dim})
            parameters = distribution.join_parameters(log_probs=torch.randn(1, 1), means=torch.randn(1, 1, dim), hidden_ik=distribution._parametrization.ipositive(1 / torch.rand(1, 1, 1)))
            sample = torch.randn(1000, dim)
            with torch.no_grad():
                pdfs = distribution.logpdf(parameters, sample).exp()
            surface = 2 * np.pi ** (dim / 2) / sci.special.gamma(dim / 2)
            integral = pdfs.mean().item() * surface
            self.assertAlmostEqual(integral, 1, delta=0.2)

    def test_pdf_product(self):
        """   """
        for d in [2, 3]:
            distribution = VMFDistribution(config={'dim': d})
            PARAMETERS1 = torch.randn(1, 3, distribution.num_parameters)
            parameters2 = torch.randn(1, 3, distribution.num_parameters)
            (prod_distribution, prod_parameters) = distribution.pdf_product(PARAMETERS1, parameters2)
            points = torch.randn(2, 3, distribution.dim)
            logpdf_g_t = distribution.logpdf(PARAMETERS1, points) + distribution.logpdf(parameters2, points)
            logpdf = prod_distribution.logpdf(prod_parameters, points)
            points0 = torch.zeros(distribution.dim)
            logpdf0_gt_ = distribution.logpdf(PARAMETERS1, points0) + distribution.logpdf(parameters2, points0)
            logpdf_g_t = logpdf_g_t - logpdf0_gt_
            logpp = prod_distribution.logpdf(prod_parameters, points0)
            logpdf = logpdf - logpp
            self.assertTrue(logpdf.allclose(logpdf_g_t, atol=1e-06))

    def test_(self):
        distribution = VMFDistribution(config={'dim': 2})
        parameters = distribution.join_parameters(log_probs=torch.tensor([[0.25]]).log(), means=torch.tensor([[[-2, 0]]]).float(), hidden_ik=distribution._parametrization.ipositive(torch.tensor([[[0.5]]])))
        with torch.no_grad():
            mls_gt = distribution.logmls(parameters, parameters).exp().item()
            (sample, _) = distribution.sample(parameters, [1000000])
            ml = distribution.logpdf(parameters, sample).exp().mean().item()
        self.assertAlmostEqual(ml, mls_gt, places=3)
if __name__ == '__main__':
    main()
