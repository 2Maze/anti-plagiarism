import itertools
import math
from unittest import TestCase, main
import numpy as np
import torch
from probabilistic_embeddings.layers.distribution.common import auto_matmul

class TestCommon(TestCase):
    """                    """

    def test_auto_matmul(self):

        def _check_case(shape_1, s):
            with torch.no_grad():
                m1 = torch.randn(*shape_1)
                m2 = torch.randn(*s)
                gt = torch.matmul(m1, m2).numpy()
                result = auto_matmul(m1, m2).numpy()
            self.assertTrue(np.allclose(result, gt, atol=1e-06))
        _check_case([0, 1], [1, 2])
        _check_case([5, 1, 4, 1, 7], [2, 4, 7, 2])
        _check_case([3, 2, 1, 4, 5], [1, 6, 5, 1])
if __name__ == '__main__':
    main()
