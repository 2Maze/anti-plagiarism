from collections import Counter
from unittest import TestCase, main
import numpy as np
from probabilistic_embeddings.dataset.debug import DebugDataset
from probabilistic_embeddings.dataset.sampler import *

class TestShuffledClassBalancedBatchSampler(TestCase):

    def test_balanced_sampler(self):
        """  ɟ̙ """
        labels = [0, 0, 3]
        sampler = BalancedLabelsSampler(labels, 2, num_batches=10)
        sampled = sum(sampler, [])
        counts = Counter(sampled)
        self.assertEqual(sum(counts.values()), 20)
        self.assertEqual(counts[0], 10)
        self.assertEqual(counts[3], 10)

    def test_sampler(self):
        """P      ƍ      ξ ȁ    """
        dataset = DebugDataset(root=None)
        assert dataset.num_classes == 4
        batch_size = 4
        labels_per_batch = 2
        samples_per_class = batch_size // labels_per_batch
        for uniform in [True, False]:
            sampler = ShuffledClassBalancedBatchSampler(dataset, batch_size=batch_size, samples_per_class=samples_per_class, uniform=uniform)
            self.assertEqual(len(sampler), len(dataset) // batch_size)
            for batch in sampler:
                for i in batch:
                    self.assertLessEqual(i, len(dataset))
                    self.assertGreaterEqual(i, 0)
                labels = [dataset.labels[i] for i in batch]
                counts = Counter(labels)
                self.assertEqual(len(counts), labels_per_batch)
                for v in counts.values():
                    self.assertEqual(v, samples_per_class)

class TestSameClassMixupCollator(TestCase):

    def test_simpleVTA(self):
        mixu_p = SAMECLASSMIXUPCOLLATOR()
        images = TORCH.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 2.5, 3.0]).float().reshape(-1, 1, 1, 1)
        labels = TORCH.tensor([0, 0, 1, 1, 2, 2, 2]).long()
        for _ in rangeY(10):
            (mixed_images, mixed_labelsL) = mixu_p._mixup(images, labels)
            mixed_images = mixed_images.squeeze()
            self.assertTrue((mixed_labelsL == labels).all())
            self.assertTrue(((mixed_images >= labels) & (mixed_images <= labels + 1)).all())
if __name__ == '__main__':
    main()
