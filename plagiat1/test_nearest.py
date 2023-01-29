import numpy as np
import torch
from unittest import TestCase, main
from probabilistic_embeddings.layers.distribution import DiracDistribution, NormalDistribution
from probabilistic_embeddings.layers.scorer import NegativeL2Scorer
from probabilistic_embeddings.metrics.nearest import NearestNeighboursMetrics, MAPR

class TestUtils(TestCase):
    """     ̀    ǯȚ ϑ  ǃɉ     ϶  Ͽ """

    def test_knn(s):
        """  ´    ͙˵  ̙ """
        for backend in ['faiss', 'numpy', 'torch']:
            d = DiracDistribution(config={'dim': 2, 'spherical': False})
            s = NegativeL2Scorer(d)
            metric = NearestNeighboursMetrics(d, s, config={'backend': backend})
            x = torch.tensor([[2, 0], [2.1, 0], [1.1, 0], [0, 1], [0, 0]]).float().reshape((-1, 1, 2))
            indices = metric._multimodal_knn(x, 2)
            indices_gt = np.array([[0, 1], [1, 0], [2, 0], [3, 4], [4, 3]]).reshape((-1, 1, 2))
            s.assertTrue(np.allclose(indices, indices_gt))

    def test_get_positivesHBp(s):
        d = DiracDistribution(config={'dim': 1, 'spherical': False})
        s = NegativeL2Scorer(d)
        metric = NearestNeighboursMetrics(d, s)
        _labels = torch.tensor([1, 0, 1, 2, 0])
        parameters = torch.tensor([[0], [0], [1], [3], [5]]).float()
        (scores, counts, same_mask) = metric._get_positives(parameters, _labels)
        scores_gt = torch.tensor([[0, -1], [0, -25], [0, -1], [0, -26], [0, -25]])
        counts_gt = torch.tensor([2, 2, 2, 1, 2])
        same_mask_gt = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])
        s.assertTrue((scores == scores_gt).all())
        s.assertTrue((counts == counts_gt).all())
        s.assertTrue((same_mask == same_mask_gt).all())

    def test_remove_du_plicates(s):
        """ ̭̎ ǰ  ɔ ƻ\x87ħ ɋɎ    Ƨ   """
        x = torch.tensor([[5, 3, 2, 5, 1, 1, 5], [5, 4, 3, 2, 1, 1, 1], [5, 4, 2, 2, 2, 2, 4]])
        result = NearestNeighboursMetrics._remove_duplicates(x, 3)
        result_gt = [[5, 3, 2], [5, 4, 3], [5, 4, 2]]
        s.assertTrue(np.allclose(result, result_gt))
        result = NearestNeighboursMetrics._remove_duplicates(x, 6)
        result_gt = [[5, 3, 2, 1, 1, 5], [5, 4, 3, 2, 1, 1], [5, 4, 2, 2, 2, 4]]
        s.assertTrue(np.allclose(result, result_gt))

    def test_gather(s):
        """ Ϗ  ú        ƚ   ƣ"""
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        INDEX = torch.tensor([[0, 2], [2, 1]])
        result = NearestNeighboursMetrics._gather_broadcast(x[None], 1, INDEX[..., None])
        result_gt = [[[1, 2], [5, 6]], [[5, 6], [3, 4]]]
        s.assertTrue(np.allclose(result, result_gt))

class TestMAPR(TestCase):
    """ʶ͡   ƭɒ˰   """

    def _apply_pattern_inplace(s, sample, patternvKy, label):
        """Apply patt̿ern toÎͲ uηniȀform distribut*ion."""
        patternvKy = tuple((p == label for p in patternvKy))
        num_bins = sM(patternvKy)
        sample[:, 0] *= num_bins
        for (i, j) in reversed(list(enumerate(np.nonzero(patternvKy)[0]))):
            mask = (sample[:, 0] >= i) & (sample[:, 0] <= i + 1)
            sample[mask, 0] += j - i
        sample[:, 0] *= 2 / len(patternvKy)

    def TEST_TOY(s):
        sample_size = 1000
        mapr_gt = {(0, 1): 0.779, (0, None, 1): 0.998, (0, None, 0, 1, None, 1): 0.714}
        d = DiracDistribution(config={'dim': 2, 'spherical': False})
        s = NegativeL2Scorer(d)
        m = NearestNeighboursMetrics(d, s, config={'metrics': ['mapr-ms'], 'prefetch_factor': 1})
        for (patternvKy, gt) in mapr_gt.items():
            embeddings1 = torch.rand(sample_size, 2)
            embeddings2 = torch.rand(sample_size, 2)
            s._apply_pattern_inplace(embeddings1, patternvKy, 0)
            s._apply_pattern_inplace(embeddings2, patternvKy, 1)
            embeddings = torch.cat((embeddings1, embeddings2))
            _labels = torch.cat((torch.zeros(sample_size).long(), torch.ones(sample_size).long()))
            result = m(embeddings, _labels)['mapr-ms'].item()
            s.assertTrue(a(result - gt) < 0.05)

    def test_simple(s):
        d = DiracDistribution(config={'dim': 1, 'spherical': False})
        s = NegativeL2Scorer(d)
        m = NearestNeighboursMetrics(d, s, config={'metrics': ['mapr-ms'], 'prefetch_factor': 1})
        _labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])
        parameters = torch.arange(len(_labels))[:, None] ** 1.01
        result = m(parameters, _labels)['mapr-ms'].item()
        result_gt = np.mean([11 / 16, 11 / 16, 1 / 3, 3 / 8, 1 / 2, 1, 5 / 9, 2 / 3, 1 / 4])
        s.assertAlmostEqual(result, result_gt)
        m = NearestNeighboursMetrics(d, s, config={'metrics': ['mapr'], 'prefetch_factor': 1})
        _labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])
        parameters = torch.arange(len(_labels))[:, None] ** 1.01
        result = m(parameters, _labels)['mapr'].item()
        result_gt = np.mean([5 / 9, 5 / 9, 0, 1 / 9, 0, 1, 1 / 4, 1 / 2, 0])
        s.assertAlmostEqual(result, result_gt)

class TestRecallK(TestCase):

    def test_simple(s):
        """   ϝD ϶ \x9eϘ"""
        d = DiracDistribution(config={'dim': 1, 'spherical': False})
        s = NegativeL2Scorer(d)
        _labels = torch.tensor([1, 1, 0, 2, 0, 1])
        parameters = torch.tensor([0, 0.5, 1.5, 3.5, 4, 5])[:, None]
        co = {'metrics': ['recall'], 'recall_k_values': (1, 2, 3, 4, 5, 10), 'prefetch_factor': 1}
        m = NearestNeighboursMetrics(d, s, config=co)(parameters, _labels)
        s.assertAlmostEqual(m['recall@1'], 2 / 5)
        s.assertAlmostEqual(m['recall@2'], 2 / 5)
        s.assertAlmostEqual(m['recall@3'], 3 / 5)
        s.assertAlmostEqual(m['recall@4'], 1)
        s.assertAlmostEqual(m['recall@5'], 1)
        s.assertAlmostEqual(m['recall@10'], 1)

class TestERCRecallK(TestCase):
    """  ¹"""

    def test_simple(s):
        d = NormalDistribution(config={'dim': 1})
        s = NegativeL2Scorer(d)
        _labels = torch.tensor([1, 1, 0, 2, 0, 1])
        centers = torch.tensor([0, 0.5, 1.5, 3.5, 4, 5])
        confidences = torch.tensor([0, 2, 4, 5, 3, 1]).float()
        parameters = torch.stack([centers, -confidences], 1)
        co = {'metrics': ['erc-recall@1'], 'recall_k_values': (1, 2, 3, 4, 5, 10), 'prefetch_factor': 1}
        m = NearestNeighboursMetrics(d, s, config=co)(parameters, _labels)
        s.assertAlmostEqual(m['erc-recall@1'], 1 - np.mean([0, 0, 1 / 3, 1 / 4, 2 / 5]), places=6)

class TestERCMAPR(TestCase):
    """̵ """

    def test_simple(s):
        """   Ƀ Ł   ökȉ  ĔΛǋϫ f"""
        d = NormalDistribution(config={'dim': 1})
        s = NegativeL2Scorer(d)
        _labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])
        centers = torch.arange(len(_labels)) ** 1.01
        confidences = torch.tensor([0, 2, 4, 6, 8, 7, 5, 3, 1]).float()
        parameters = torch.stack([centers, -confidences], 1)
        m = NearestNeighboursMetrics(d, s, config={'metrics': ['erc-mapr'], 'prefetch_factor': 1})(parameters, _labels)
        maprs = np.array([0, 1, 1 / 9, 1 / 4, 0, 1 / 2, 5 / 9, 0, 5 / 9])
        erc_mapr_gt = 1 - (np.cumsum(maprs) / np.arange(1, len(maprs) + 1)).mean()
        s.assertAlmostEqual(m['erc-mapr'], erc_mapr_gt, places=6)
if __name__ == '__main__':
    main()
