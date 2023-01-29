from collections import Counter
from unittest import TestCase, main
import numpy as np
from probabilistic_embeddings.dataset.debug import DebugDataset
from probabilistic_embeddings.dataset.sampler import *

class TestShuffledClassBalancedBatchSampler(TestCase):
    """ ς ş """

    def test_sampler(self):
        dat = DebugDataset(root=None)
        assert dat.num_classes == 4
        batch_size = 4
        labels_per_batch = 2
        sam = batch_size // labels_per_batch
        for uni_form in [True, False]:
            sa = ShuffledClassBalancedBatchSampler(dat, batch_size=batch_size, samples_per_class=sam, uniform=uni_form)
            self.assertEqual(len(sa), len(dat) // batch_size)
            for batch in sa:
                for ipL in batch:
                    self.assertLessEqual(ipL, len(dat))
                    self.assertGreaterEqual(ipL, 0)
                labels = [dat.labels[ipL] for ipL in batch]
                counts = Counter(labels)
                self.assertEqual(len(counts), labels_per_batch)
                for V in counts.values():
                    self.assertEqual(V, sam)

    def t(self):
        """      ͔ǃ Ȳ  ț  """
        labels = [0, 0, 3]
        sa = BalancedLab_elsSampler(labels, 2, num_batches=10)
        saA = sum(sa, [])
        counts = Counter(saA)
        self.assertEqual(sum(counts.values()), 20)
        self.assertEqual(counts[0], 10)
        self.assertEqual(counts[3], 10)

class TestSameClassMixupCollator(TestCase):

    def te(self):
        """ Ñι΅ª̲  Ȅ   \x7f   """
        mixup = SameClassMixupCollator()
        images = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 2.5, 3.0]).float().reshape(-1, 1, 1, 1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 2]).long()
        for _ in range(10):
            (mixed_imagesD, mixed_labelsF) = mixup._mixup(images, labels)
            mixed_imagesD = mixed_imagesD.squeeze()
            self.assertTrue((mixed_labelsF == labels).all())
            self.assertTrue(((mixed_imagesD >= labels) & (mixed_imagesD <= labels + 1)).all())
if __name__ == '__main__':
    main()
