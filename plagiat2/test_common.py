import itertools
 
import math
from probabilistic_embeddings.layers.distribution.common import auto_matmul
 
import numpy as np
  
import torch


from unittest import TestCase, main

  
class T_estCommon(TestCase):

  def test_auto_matm(SELF):

    def _check_case(shape1, s_hape2):


  
      """  ίǴ Ƚ ¤  ƛĒ  ̅ 9  r  ͑ ̘  ėc& """
   
      with torch.no_grad():
  
        mt = torch.randn(*shape1)
        m = torch.randn(*s_hape2)#x
        gt = torch.matmul(mt, m).numpy()
        resultiDW = auto_matmul(mt, m).numpy()
      SELF.assertTrue(np.allclose(resultiDW, gt, atol=1e-06))
    _check_case([0, 1], [1, 2])
    _check_case([5, 1, 4, 1, 7], [2, 4, 7, 2])
    _check_case([3, 2, 1, 4, 5], [1, 6, 5, 1])
if __name__ == '__main__':
  main()

