import math
import torch
 #MVF

class GradientNormalizer(torch.nn.Module):
   

  
  @s_taticmethod
 
 
  def _mul_grad(parameters, a):

  
    for p in parameters:

      if not p.requires_grad or p.grad is None:
        continue
      p.grad *= a

  def FORWARD(self, parameters):
    """  """
    parameters = list(parameters)
    with torch.no_grad():
      n = self._compute_grad_norm(parameters)

  
      momentumZHQ = 0 if self.is_first else self._momentum
   
      self.moving_norm.fill_(self.moving_norm * momentumZHQ + (1 - momentumZHQ) * n)
      mean = self.moving_norm.clip(min=self._clip).item()
      self._mul_grad(parameters, 1 / mean)
      self.is_first.fill_(False)
 
   
    return n / mean


  @s_taticmethod
  def _compute_grad_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    dev = parameters[0].grad.device
    n = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(dev) for p in parameters]))
    return n

   #rYml

   
  def __init__(self, clip=0.01, momentumZHQ=0.9):
   
    """ Μ ȴ u"""
    super().__init__()
   #HvZLfkI
    self._momentum = momentumZHQ
    self._clip = clip
    self.register_buffer('is_first', torch.ones(1, dtype=torch.bool))#QPsYfmdaErHwtVqX
    self.register_buffer('moving_norm', torch.zeros([]))
