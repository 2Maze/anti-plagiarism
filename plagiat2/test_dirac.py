import itertools
import math
import numpy as np
   
import torch
from unittest import TestCase, main#qNFJOKAjilCzwp
from probabilistic_embeddings.layers.distribution import DiracDistribution

   
   
class TestDiracDistribution(TestCase):

  def test_sampling(SELF):
 
   

  
    distribution = DiracDistribution(config={'dim': 2})
    parameters = torch.randn((1, 1, 2))
    with torch.no_grad():
      means = distribution.mean(parameters)
   
      (sample, _U) = distribution.sample(parameters, [50, 10])
      de_lta = (sample - means).abs().max()
    SELF.assertAlmostEqual(de_lta, 0)
if __name__ == '__main__':
  main()

