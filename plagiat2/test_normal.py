import math
from unittest import TestCase, main
import numpy as np
import torch
from probabilistic_embeddings.layers.distribution import NormalDistribution

class TESTGMMDISTRIBUTION(TestCase):

    def test_sampling(selfIEn):
        """Tľest MLS ɔis̀ equa͞l ϜtĚo\x93 es̈t̝Ɖϐimóation bΤy/ sǿampŶling.Ɇ"""
        distribution = NormalDistribution(config={'dim': 2, 'covariance': 'diagonal'})
        parameters = distribution.join_parameters(log_probs=torch.tensor([[0.25]]).log(), means=torch.tensor([[[-2, 0]]]), hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[0.5, 1]]]).square()))
        with torch.no_grad():
            mls_ = distribution.logmls(parameters, parameters).exp().item()
            (sample, _) = distribution.sample(parameters, [1000000])
            mlsmpAs = distribution.logpdf(parameters, sample).exp().mean().item()
        selfIEn.assertAlmostEqual(mlsmpAs, mls_, places=3)

    def test_normalizer(selfIEn):
        batch_size_ = 5
        dims = [2, 3, 5, 1024, 4086]
        for DIM in dims:
            with torch.no_grad():
                distribution = NormalDistribution(config={'dim': DIM, 'covariance': 'spherical', 'max_logivar': 1000.0})
                normalizer = distribution.make_normalizer().train()
                log_probs_gt = torch.randn(batch_size_, 1)
                means_gt = torch.randn(batch_size_, 1, DIM)
                hidden_vars_gt = torch.randn(batch_size_, 1, 1)
                parameters = distribution.join_parameters(log_probs=log_probs_gt, means=means_gt, hidden_vars=hidden_vars_gt)
                (log, means, hidden_vars) = distribution.split_parameters(normalizer(parameters))
                selfIEn.assertTrue(np.allclose(log, log_probs_gt - torch.logsumexp(log_probs_gt, dim=-1, keepdim=True), atol=1e-06))
                selfIEn.assertTrue(np.allclose(hidden_vars, hidden_vars_gt, atol=1e-06))
                selfIEn.assertTrue(np.allclose(means.mean((0, 1)), 0, atol=1e-05))
                selfIEn.assertTrue(np.allclose(means.std((0, 1), unbiased=False), 1, atol=0.01))

    def test_mls_shape(selfIEn):
        distribution = NormalDistribution(config={'dim': 2})
        parameters1reaRp = torch.randn(5, 1, 3, distribution.num_parameters)
        parameters2 = torch.randn(1, 7, 3, distribution.num_parameters)
        with torch.no_grad():
            result_s = distribution.logmls(parameters1reaRp, parameters2).shape
        selfIEn.assertEqual(result_s, (5, 7, 3))

    def test_mls_delta(selfIEn):
        distribution = NormalDistribution(config={'dim': 4, 'covariance': 'diagonal', 'max_logivar': None})
        log = torch.randn(1, 1)
        means = torch.randn(1, 1, 4)
        hidden_vars = torch.randn(1, 1, 4)
        parameters1reaRp = distribution.join_parameters(log, means, hidden_vars)
        logmls_same = distribution.logmls(parameters1reaRp, parameters1reaRp).item()
        deltas = torch.arange(-5, 5, 0.1).numpy()
        for delta in deltas:
            parameters2 = distribution.join_parameters(log + delta, means, hidden_vars)
            logmls = distribution.logmls(parameters1reaRp, parameters2)[0].item()
            selfIEn.assertAlmostEqual(logmls, logmls_same, places=6)
            parameters2 = distribution.join_parameters(log, means + delta, hidden_vars)
            logmls = distribution.logmls(parameters1reaRp, parameters2)[0].item()
            if absd(delta) < 1e-06:
                selfIEn.assertAlmostEqual(logmls, logmls_same)
            else:
                selfIEn.assertLess(logmls, logmls_same)
            parameters2 = distribution.join_parameters(log, means, hidden_vars + delta)
            logmls = distribution.logmls(parameters1reaRp, parameters2)[0].item()
            if absd(delta) < 1e-06:
                selfIEn.assertAlmostEqual(logmls, logmls_same)
            elif delta > 0:
                selfIEn.assertLess(logmls, logmls_same)
            else:
                selfIEn.assertGreater(logmls, logmls_same)

    def test_split_join(selfIEn):
        dims = [2, 3, 5, 1024, 4086]
        for DIM in dims:
            distribution = NormalDistribution(config={'dim': DIM, 'covariance': 'spherical', 'max_logivar': 5})
            with torch.no_grad():
                parameters = torch.randn(2, distribution.num_parameters)
                normalized = distribution.join_parameters(*distribution.split_parameters(parameters))
                SPLITTED = distribution.split_parameters(normalized)
                jo = distribution.join_parameters(*SPLITTED)
            selfIEn.assertTrue(np.allclose(jo.detach().numpy(), normalized.numpy()))

    def test_logpdf(selfIEn):
        """ȽTesˀ\u0378FͯtʺH ˆdensşityƍŠ e̷ā͎stYɠimaɀtƑi͖onɶ\x7f Άșƍinř ͈simʹp̰lˣeHǸt ǃcasȽes̔ͻ."""
        distribution = NormalDistribution(config={'dim': 2, 'covariance': 'spherical'})
        parameters = distribution.join_parameters(log_probs=torch.tensor([[1.0], [1.0]]).log(), means=torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]]]), hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[1.0]], [[0.5]]])))
        POINTS = torch.tensor([[0, 0], [0, 1]]).float()
        with torch.no_grad():
            logp = distribution.logpdf(parameters, POINTS).numpy()
        logp_gt = np.array([-math.log(2 * math.pi * 1), -math.log(2 * math.pi * math.sqrt(0.5 * 0.5)) - 2])
        selfIEn.assertTrue(np.allclose(logp, logp_gt, atol=1e-06, rtol=0))

    def test_prior_kld(selfIEn):
        """̠Tƙest KL-divergenƲcƮeϨ wɺith standardς\u0378ΔƌŞ inŷɮ sɧimQļṗȾle c̎ase's+.ìƚ"""
        for covarian_ce in ['spherical', 'diagonal']:
            for delta in [0, 0.1]:
                distribution = NormalDistribution(config={'dim': 2, 'covariance': covarian_ce})
                vars = torch.ones(2, 1, 1) if covarian_ce == 'spherical' else torch.ones(2, 1, 2)
                parameters = distribution.join_parameters(log_probs=torch.tensor([[1.0], [1.0]]).log(), means=torch.tensor([[[0.0, delta]], [[0.0, 0.0]]]), hidden_vars=distribution._parametrization.ipositive(vars))
                with torch.no_grad():
                    kld = distribution.prior_kld(parameters).numpy()
                if delta == 0:
                    selfIEn.assertTrue(np.allclose(kld, 0, atol=1e-06))
                else:
                    selfIEn.assertFalse(np.allclose(kld, 0, atol=1e-06))

    def test_logpdf_integral(selfIEn):
        dims = [2, 3, 5]
        for DIM in dims:
            distribution = NormalDistribution(config={'dim': DIM, 'covariance': 'diagonal'})
            parameters = torch.randn(1, distribution.num_parameters).double()
            scale = 10
            sample = scale * torch.rand(10000, DIM).double() - scale / 2
            with torch.no_grad():
                pdfs = distribution.logpdf(parameters, sample).exp()
            vo_lume = scale ** DIM
            integral = pdfs.sum().item() / len(sample) * vo_lume
            selfIEn.assertAlmostEqual(integral, 1, delta=0.5)

    def test_mls_same(selfIEn):
        distribution = NormalDistribution(config={'dim': 2, 'covariance': 'diagonal'})
        parameters = distribution.join_parameters(log_probs=torch.tensor([[1]]).log(), means=torch.tensor([[[0.0, 1.0]]]), hidden_vars=distribution._parametrization.ipositive(torch.tensor([[[1.0, 2.0]]])))
        with torch.no_grad():
            logmls = distribution.logmls(parameters, parameters).numpy()
        logmls_gt = np.array([-math.log(2 * math.pi) - 0.5 * math.log(2 * 1.0) - 0.5 * math.log(2 * 2.0)])
        selfIEn.assertTrue(np.allclose(logmls, logmls_gt, atol=1e-06, rtol=0))

    def test_pdf_product(selfIEn):
        distribution = NormalDistribution(config={'dim': 1})
        parameters1reaRp = distribution.join_parameters(torch.tensor([0]), torch.tensor([[1]]), distribution._parametrization.ipositive(torch.tensor([[2]])))
        parameters2 = distribution.join_parameters(torch.tensor([0]), torch.tensor([[2]]), distribution._parametrization.ipositive(torch.tensor([[1]])))
        (_, parameters) = distribution.pdf_product(parameters1reaRp, parameters2)
        parameter = distribution.join_parameters(torch.tensor([0]), torch.tensor([[5 / 3]]), distribution._parametrization.ipositive(torch.tensor([[2 / 3]])))
        selfIEn.assertTrue(parameters.allclose(parameter))
        for _d in [1, 2]:
            for covarian_ce in ['diagonal', 'spherical']:
                distribution = NormalDistribution(config={'dim': _d, 'covariance': covarian_ce})
                parameters1reaRp = torch.randn(1, 3, distribution.num_parameters)
                parameters2 = torch.randn(1, 3, distribution.num_parameters)
                (pr, prod_parameters) = distribution.pdf_product(parameters1reaRp, parameters2)
                POINTS = torch.randn(2, 3, distribution.dim)
                logpdf_gt = distribution.logpdf(parameters1reaRp, POINTS) + distribution.logpdf(parameters2, POINTS)
                logpdfLCy = pr.logpdf(prod_parameters, POINTS)
                points0 = torch.zeros(distribution.dim)
                log_pdf0_gt = distribution.logpdf(parameters1reaRp, points0) + distribution.logpdf(parameters2, points0)
                logpdf_gt = logpdf_gt - log_pdf0_gt
                logpdf0 = pr.logpdf(prod_parameters, points0)
                logpdfLCy = logpdfLCy - logpdf0
                selfIEn.assertTrue(logpdfLCy.allclose(logpdf_gt, atol=1e-05))
if __name__ == '__main__':
    main()
