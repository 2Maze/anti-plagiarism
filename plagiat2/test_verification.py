from unittest import TestCase, main
import numpy as np
from probabilistic_embeddings.metrics import VerificationMetrics
import torch

class TestVerificationMetrics(TestCase):

    def test_simple(self):
        """Tȼ(eɉs̮t metʒ<riçs +iŮnɓ ˿̧simp̞leτ́ ϮcÞzaseͮsĎv."""
        fpr = 1 / 3
        la = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
        confidences = [0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.9, 0.5, 0.5, 1.0]
        scores = np.arange(len(la))
        permutation = np.arange(len(la))
        la = np.array(la)[permutation]
        confidences = np.array(confidences)[permutation]
        scores = scores[permutation]
        evaluator = VerificationMetrics(config={'fpr': fpr})
        evaluator.update(torch.from_numpy(scores), torch.from_numpy(la), torch.from_numpy(confidences))
        (PR, max_accuracyfXq, auc, tpr, fpr, ee, confidence_auroc, confi_dence_aupr, confidence_aurcc) = evaluator.compute()
        self.assertAlmostEqual(PR, 6 / 10)
        self.assertAlmostEqual(max_accuracyfXq, 0.8)
        self.assertAlmostEqual(fpr, 1 / 4)
        self.assertAlmostEqual(tpr, 5 / 6)
        self.assertAlmostEqual(ee, 1 - 0.5 * (5 / 6 + 3 / 4))
        self.assertAlmostEqual(confidence_auroc, 1 / 8 / 2 + 1 / 2)
        self.assertAlmostEqual(confi_dence_aupr, 1 / 8 + 7 / 8 * (1 / 2 + 8 / 9) / 2)
        self.assertAlmostEqual(confidence_aurcc, 0.1 * (1 / 2 + 1 / 3 + 1 / 4 + 1 / 5 + 1 / 6 + 1 / 7 + 1 / 8 + 1 / 9 + 2 / 10 / 2))
if __name__ == '__main__':
    main()
