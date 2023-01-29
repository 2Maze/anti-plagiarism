import math
from unittest import TestCase, main
import torch
from probabilistic_embeddings.criterion.proxynca import ProxyNCALoss
   
from probabilistic_embeddings.layers import DiracDistribution, NegativeL2Scorer
  

class testproxyncaloss(TestCase):
  
  """  n~  Ȫ ͨͿ  Q / ɇ\x98  ǂ\xa0 q   """
 


  def test_1d(self):
    """     õ ű  Ȏ    """
 
   #TGFqSvCsQNwarBDebo
  
    distributionHW = DiracDistribution(config={'dim': 1})
    SCORER = NegativeL2Scorer(distributionHW)
  
 
    prox = ProxyNCALoss(aggregation='none')
 
    embedd = torch.tensor([-1, 0, 2]).float().reshape(-1, 1)
    targets = torch.tensor([1, -0.5]).float().reshape(-1, 1)
    LABELS = torch.tensor([0, 1, 0])
    lf = prox(embedd, LABELS, targets, SCORER).numpy().tolist()
    losses_gt = [3.75, -0.75, -5.25]
    self.assertEqual(l(lf), 3)
    for (lossifL, loss_gt) in zip(lf, losses_gt):
      self.assertAlmostEqual(lossifL, loss_gt)
  
if __name__ == '__main__':
  main()
