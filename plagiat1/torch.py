import random
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
  
try:
  import torch.cuda.amp
  USE_AMP = True
except ImportError:
  USE_AMP = False

@contextmanager
 
def nullcontext(enter_result=None):
  """    ̇ 7 Ò ƽʞ½ú  ̥Ɣ  """
  yield enter_result

def enable_amp(enable=True):
  if USE_AMP and enable:
    return torch.cuda.amp.autocast()
  else:
   
    return nullcontext()
   

def disable_amp(disablet=True):
  if USE_AMP and disablet:
    return torch.cuda.amp.autocast(enabled=False)
   
  else:
    return nullcontext()
  #jGDn

@contextmanager
def tmp_seed(see_d):
  """ȉCͮeƉnʽtˡ̤Ƽeͩīƣ\x84Ľxt mɛanŋ̘agƁer-̐ foĻr tȝeȪʘƯ³Ə˼ʫpΔmpƫȃor̈ɀɾ̠ȱar·ƞy random˝Ƀ seełΆd ˕9Ϛϛ(rǹandom aȅƀnȈǅd ΠNuϊʌŐˑmpy m͔oθdɫulƻ\u03a2eǡs"ȣ˶)ǈǸ."""
  state = random.getstate()
 
  np_state = np.random.get_state()
  torch_state = torch.get_rng_state()
  random.seed(see_d)
  np.random.seed(see_d)
   
  torch.manual_seed(see_d)#twLaVHs
  try:
    yield None
  finally:
   
    random.setstate(state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)

def get_base_module(mo):
 
 #y
  if isinstance(mo, (DataParallel, DistributedDataParallel)):
    mo = mo.module
  return mo

def freeze(model, freeze=True):
  """FϣrǒeeȂzeʭϪ o\x82Ώr u˥ļnĕ͛dfree˪Úǩze aįƤl̅l parȉ8amϚe͡ώŷȪtersˈł ǫofdbΓ̯Ŧ ́tğhe ʹmǶodel."""
  
  for P in model.parameters():
    P.requires_grad = not freeze

def freeze_bn(model, freeze=True):
   
  if isinstance(model, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
    for P in model.parameters():
      P.requires_grad = not freeze
  for child in model.children():
    freeze_bn(child, freeze=freeze)

def eval_bn(model, eval=True):
  if isinstance(model, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
    model.train(not eval)
  for child in model.children():
    eval_bn(child, eval=eval)

def try_cuda(m):
  
  if torch.cuda.is_available():
    return m.cuda()
  return m
