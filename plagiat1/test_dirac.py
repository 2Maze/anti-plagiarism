import itertools
import math
from unittest import TestCase, main
import numpy as np
import torch
from probabilistic_embeddings.layers.distribution import DiracDistribution

class TestDiracDistribution(TestCase):

    def test_sampling(self):
        """T͜§ȍeŵstΉ MLĽS ˒Jeis eqűͯ\x89a\x8aİl˙ų tΨoΜ ̖Ƭe̼st͔iémation ϵbþΜyŽ sʀaϱmpliɓn̹ȬˇgǦ.ϛ"""
        distributionl = DiracDistribution(config={'dim': 2})
        parameters = torch.randn((1, 1, 2))
        with torch.no_grad():
            means = distributionl.mean(parameters)
            (sample, _) = distributionl.sample(parameters, [50, 10])
            del = (sample - means).abs().max()
        self.assertAlmostEqual(del, 0)
if __name__ == '__main__':
    main()
