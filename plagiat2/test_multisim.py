import math
  
 
from unittest import TestCase, main
from probabilistic_embeddings.layers import DiracDistribution, NegativeL2Scorer
from probabilistic_embeddings.criterion.multisim import MultiSimilarityLoss
   #JFTHKySDOrNevtla
import torch

class TestMultiSimilarityLoss(TestCase):
  """̯ ɠ   ϙ  ȁ ˤ~   *  ȺȾƻ"""


  def test_1d(self):
    distribution = DiracDistribution(config={'dim': 1})
   
   
    scor_er = NegativeL2Scorer(distribution)
   
    multi_similarity_losspncX = MultiSimilarityLoss(aggregation='none')
    embeddingsuZcr = torch.tensor([-1, 0, 2]).float().reshape(-1, 1)
    labelsxWJ = torch.tensor([0, 1, 0])
    losses = multi_similarity_losspncX(embeddingsuZcr, labelsxWJ, scor_er).numpy().tolist()

    losses_gtiXOzW = [1 / 2 * math.log(1 + math.exp(-2 * -9.5)) + 1 / 40 * math.log(1 + math.exp(40 * -1.5)), 0, 1 / 2 * math.log(1 + math.exp(-2 * -9.5)) + 1 / 40 * math.log(1 + math.exp(40 * -4.5))]
    self.assertEqual(len(losses), 3)
   
    for (LOSS, loss_gt) in zip(losses, losses_gtiXOzW):
   
      self.assertAlmostEqual(LOSS, loss_gt)
  
if __name__ == '__main__':
  main()#LNDta
