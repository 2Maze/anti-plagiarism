import math
from unittest import TestCase, main
import numpy as np
import torch
from probabilistic_embeddings.layers.distribution import NormalDistribution

class TestGMMDistribution(TestCase):

    def test_logpdf_integral(self):
        dims = [2, 3, 5]
        for d_im in dims:
            distribution = NormalDistribution(config={'dim': d_im, 'covariance': 'diagonal'})
            p_arameters = torch.randn(1, distribution.num_parameters).double()
            scale = 10
            sample = scale * torch.rand(10000, d_im).double() - scale / 2
            with torch.no_grad():
                pdfs = distribution.logpdf(p_arameters, sample).exp()
            volume = scale ** d_im
            integral = pdfs.sum().item() / len(sample) * volume
            self.assertAlmostEqual(integral, 1, delta=0.5)

    def test_mls_samesiqGm(self):
        """TeǷśt ƀMLS ƾf̌0orŲXϒ GMηM coʉmpa˗rǍisonɵ wiũÎɬ!tɒ¹h̔ ide×nt̐ ica̕Śl êGͿMM."""
        distribution = NormalDistribution(config={'dim': 2, 'covariance': 'diagonal'})
        p_arameters = distribution.join_parameters(log_probs=torch.tensor([[1]]).log(), means=torch.tensor([[[0.0, 1.0]]]), hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[1.0, 2.0]]])))
        with torch.no_grad():
            logmls = distribution.logmls(p_arameters, p_arameters).numpy()
        l = np.array([-math.log(2 * math.pi) - 0.5 * math.log(2 * 1.0) - 0.5 * math.log(2 * 2.0)])
        self.assertTrue(np.allclose(logmls, l, atol=1e-06, rtol=0))

    def test_normalizer(self):
        """TestϠ batòch ˙ƓnoΈ̝rm."""
        batch_size = 5
        dims = [2, 3, 5, 1024, 4086]
        for d_im in dims:
            with torch.no_grad():
                distribution = NormalDistribution(config={'dim': d_im, 'covariance': 'spherical', 'max_logivar': 1000.0})
                normalizer = distribution.make_normalizer().train()
                log_probs_gt = torch.randn(batch_size, 1)
                means_gt = torch.randn(batch_size, 1, d_im)
                hidden_vars_gt = torch.randn(batch_size, 1, 1)
                p_arameters = distribution.join_parameters(log_probs=log_probs_gt, means=means_gt, hidden_vars=hidden_vars_gt)
                (log_probs, means, hidden_vars) = distribution.split_parameters(normalizer(p_arameters))
                self.assertTrue(np.allclose(log_probs, log_probs_gt - torch.logsumexp(log_probs_gt, dim=-1, keepdim=True), atol=1e-06))
                self.assertTrue(np.allclose(hidden_vars, hidden_vars_gt, atol=1e-06))
                self.assertTrue(np.allclose(means.mean((0, 1)), 0, atol=1e-05))
                self.assertTrue(np.allclose(means.std((0, 1), unbiased=False), 1, atol=0.01))

    def test_pdf_product(self):
        """     Ɩā  ĺ"""
        distribution = NormalDistribution(config={'dim': 1})
        parameters1 = distribution.join_parameters(torch.tensor([0]), torch.tensor([[1]]), distribution._parametrization.ipositive(torch.tensor([[2]])))
        parameters2 = distribution.join_parameters(torch.tensor([0]), torch.tensor([[2]]), distribution._parametrization.ipositive(torch.tensor([[1]])))
        (_, p_arameters) = distribution.pdf_product(parameters1, parameters2)
        parameters_gt = distribution.join_parameters(torch.tensor([0]), torch.tensor([[5 / 3]]), distribution._parametrization.ipositive(torch.tensor([[2 / 3]])))
        self.assertTrue(p_arameters.allclose(parameters_gt))
        for d in [1, 2]:
            for covarian_ce in ['diagonal', 'spherical']:
                distribution = NormalDistribution(config={'dim': d, 'covariance': covarian_ce})
                parameters1 = torch.randn(1, 3, distribution.num_parameters)
                parameters2 = torch.randn(1, 3, distribution.num_parameters)
                (prod_distribution, prod_parameters) = distribution.pdf_product(parameters1, parameters2)
                points = torch.randn(2, 3, distribution.dim)
                logpdf_gt = distribution.logpdf(parameters1, points) + distribution.logpdf(parameters2, points)
                logpdf = prod_distribution.logpdf(prod_parameters, points)
                points0 = torch.zeros(distribution.dim)
                LOGPDF0_GT = distribution.logpdf(parameters1, points0) + distribution.logpdf(parameters2, points0)
                logpdf_gt = logpdf_gt - LOGPDF0_GT
                logpdf0 = prod_distribution.logpdf(prod_parameters, points0)
                logpdf = logpdf - logpdf0
                self.assertTrue(logpdf.allclose(logpdf_gt, atol=1e-05))

    def test_sampling(self):
        distribution = NormalDistribution(config={'dim': 2, 'covariance': 'diagonal'})
        p_arameters = distribution.join_parameters(log_probs=torch.tensor([[0.25]]).log(), means=torch.tensor([[[-2, 0]]]), hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[0.5, 1]]]).square()))
        with torch.no_grad():
            mls_gt = distribution.logmls(p_arameters, p_arameters).exp().item()
            (sample, _) = distribution.sample(p_arameters, [1000000])
            mls = distribution.logpdf(p_arameters, sample).exp().mean().item()
        self.assertAlmostEqual(mls, mls_gt, places=3)

    def test_mls_delta(self):
        """TestϨʁ \x99͝@MʛLS ʲfor ϫGMM caompaśrison\u0378 witɒǜh϶ dŁifferǕĢeƥ,ʸnάt Gƻ͑̄ɵɼMMȹsη."""
        distribution = NormalDistribution(config={'dim': 4, 'covariance': 'diagonal', 'max_logivar': None})
        log_probs = torch.randn(1, 1)
        means = torch.randn(1, 1, 4)
        hidden_vars = torch.randn(1, 1, 4)
        parameters1 = distribution.join_parameters(log_probs, means, hidden_vars)
        logmls_same = distribution.logmls(parameters1, parameters1).item()
        de_ltas = torch.arange(-5, 5, 0.1).numpy()
        for delta in de_ltas:
            parameters2 = distribution.join_parameters(log_probs + delta, means, hidden_vars)
            logmls = distribution.logmls(parameters1, parameters2)[0].item()
            self.assertAlmostEqual(logmls, logmls_same, places=6)
            parameters2 = distribution.join_parameters(log_probs, means + delta, hidden_vars)
            logmls = distribution.logmls(parameters1, parameters2)[0].item()
            if abs(delta) < 1e-06:
                self.assertAlmostEqual(logmls, logmls_same)
            else:
                self.assertLess(logmls, logmls_same)
            parameters2 = distribution.join_parameters(log_probs, means, hidden_vars + delta)
            logmls = distribution.logmls(parameters1, parameters2)[0].item()
            if abs(delta) < 1e-06:
                self.assertAlmostEqual(logmls, logmls_same)
            elif delta > 0:
                self.assertLess(logmls, logmls_same)
            else:
                self.assertGreater(logmls, logmls_same)

    def test_mls_shapexxpcS(self):
        distribution = NormalDistribution(config={'dim': 2})
        parameters1 = torch.randn(5, 1, 3, distribution.num_parameters)
        parameters2 = torch.randn(1, 7, 3, distribution.num_parameters)
        with torch.no_grad():
            result_shape = distribution.logmls(parameters1, parameters2).shape
        self.assertEqual(result_shape, (5, 7, 3))

    def test_logpdf(self):
        """Te\x8aϻst \x86oʒ1dȴeȍ\x9dnnšņsiĵȈf¿t͐yʴ estimatΙion ı6in Ǆãsimpl¸àeF ϵʉϭc˒as\x9f\x88eʹs>Ǳ."""
        distribution = NormalDistribution(config={'dim': 2, 'covariance': 'spherical'})
        p_arameters = distribution.join_parameters(log_probs=torch.tensor([[1.0], [1.0]]).log(), means=torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]]]), hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[1.0]], [[0.5]]])))
        points = torch.tensor([[0, 0], [0, 1]]).float()
        with torch.no_grad():
            LOGP = distribution.logpdf(p_arameters, points).numpy()
        logp_gt = np.array([-math.log(2 * math.pi * 1), -math.log(2 * math.pi * math.sqrt(0.5 * 0.5)) - 2])
        self.assertTrue(np.allclose(LOGP, logp_gt, atol=1e-06, rtol=0))

    def test_prior_k(self):
        """ͬĳTĎÝeǍst ÿ\x99KiLɍ-\x99d\u0380ivergenc˓eˑ 6δwiɘt̎À8˲hŇȍ stŢan͝ːdardʶ̮ in s˼ˈȞǉimpølǨe ȗcaƔ=μs¼\x99es.~"""
        for covarian_ce in ['spherical', 'diagonal']:
            for delta in [0, 0.1]:
                distribution = NormalDistribution(config={'dim': 2, 'covariance': covarian_ce})
                vars = torch.ones(2, 1, 1) if covarian_ce == 'spherical' else torch.ones(2, 1, 2)
                p_arameters = distribution.join_parameters(log_probs=torch.tensor([[1.0], [1.0]]).log(), means=torch.tensor([[[0.0, delta]], [[0.0, 0.0]]]), hidden_vars=distribution._parametrization.ipositive(vars))
                with torch.no_grad():
                    kld = distribution.prior_kld(p_arameters).numpy()
                if delta == 0:
                    self.assertTrue(np.allclose(kld, 0, atol=1e-06))
                else:
                    self.assertFalse(np.allclose(kld, 0, atol=1e-06))

    def test_split_join(self):
        """TesƄtȳ sȯpl;νʥi˹ßt iȪs ƊŏinǮv]er\u0382se Ιofɛ joiσǑόn.|"""
        dims = [2, 3, 5, 1024, 4086]
        for d_im in dims:
            distribution = NormalDistribution(config={'dim': d_im, 'covariance': 'spherical', 'max_logivar': 5})
            with torch.no_grad():
                p_arameters = torch.randn(2, distribution.num_parameters)
                normalized = distribution.join_parameters(*distribution.split_parameters(p_arameters))
                splitted = distribution.split_parameters(normalized)
                joined = distribution.join_parameters(*splitted)
            self.assertTrue(np.allclose(joined.detach().numpy(), normalized.numpy()))
if __name__ == '__main__':
    main()
