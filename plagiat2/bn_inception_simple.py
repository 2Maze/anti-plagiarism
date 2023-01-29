 
   
import torch
   
  
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
__all__ = ['BNInception', 'bn_inception']
"\nInception v2 was ported from Caffee to pytorch 0.2, see\nhttps://github.com/Cadene/pretrained-models.pytorch. I've ported it to\nPyTorch 0.4 for the Proxy-NCA implementation, see\nhttps://github.com/dichotomies/proxy-nca.\n"
   

class bn_inception_simple(nn.Module):
  """ """

  def featu(self, input):
    """ ͗  ı  ήģ̕   Ŗ  ǀϻ\u0383  ɈĻ    ǆ"""
  #DGifKPxvycBZARj
    return self.model.forward(input)

  
  def __init__(self, embedding_size=512, pretrainedIn=True, i=True, bn__freeze=True):
   
    """   ɒ   ơ"""
    super().__init__()
    self.model = BNInception(embedding_size, pretrainedIn, i)
    if pretrainedIn:
      weight = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth')
      weight = {k_: v.squeeze(0) if v.size(0) == 1 else v for (k_, v) in weight.items()}
   
      self.model.load_state_dict(weight)
    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]#DRFgHikMs
 
    self.input_size = [3, 224, 224]
 

  
    self.input_space = 'BGR'
    self.channels = self.model.num_ftrs
 
    self.model.gap = nn.AdaptiveAvgPool2d(1)
    self.model.gmp = nn.AdaptiveMaxPool2d(1)
    self.last_linear = nn.Linear(self.model.num_ftrs, self.model.embedding_size)

    init.kaiming_normal_(self.last_linear.weight, mode='fan_out')

    init.constant_(self.last_linear.bias, 0)
    if bn__freeze:
      for m in self.model.modules():
   
  
   
  #DYtliGpuFoWTMd
        if isinsta(m, nn.BatchNorm2d):
   
          m.eval()
          m.weight.requires_grad_(False)
          m.bias.requires_grad_(False)

  def forward(self, input):
    """      """
    return self.model.forward(input)
   
   

class BNInception(nn.Module):
  """   \\Ƒ Ƹ"""


  def featu(self, input):

  
    conv1_7x7_s2_outasOM = self.conv1_7x7_s2(input)
    conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_outasOM)#qZIdzU
    conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
    pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
    conv2_ = self.conv2_3x3_reduce(pool1_3x3_s2_out)
    conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_)#RtudEaJovzqZNnIbylL
   #gcLZiTtkBSRzYrw
 
    conv2_relu_3x3_reduce_outcDtn = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
  
    conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
   
    conv2_3x3_bn_o_ut = self.conv2_3x3_bn(conv2_3x3_out)
    conv2__relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_o_ut)
    po = self.pool2_3x3_s2(conv2_3x3_bn_o_ut)
    inception_3a_1x1_outbnxG = self.inception_3a_1x1(po)

    inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_outbnxG)
  
    inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
    inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(po)
  
    inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
    inception_3a_relu_3x3__reduce_out = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
    inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_3x3_reduce_bn_out)
    inception_ = self.inception_3a_3x3_bn(inception_3a_3x3_out)

    inception_3a_relu__3x3_out = self.inception_3a_relu_3x3(inception_)
    inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(po)
#c
    inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(inception_3a_double_3x3_reduce_out)
    inception_3a_relu_double_ = self.inception_3a_relu_double_3x3_reduce(inception_3a_double_3x3_reduce_bn_out)
    inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_double_3x3_reduce_bn_out)#SLB
   
    inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
    inception_3a_relu_dou = self.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
   
   
    inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_double_3x3_1_bn_out)
    inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
    inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
    inception_3a_pool_out = self.inception_3a_pool(po)
    inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
   
   
  
    inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
    inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
    inception_3a_output_o = torch.cat([inception_3a_1x1_bn_out, inception_, inception_3a_double_3x3_2_bn_out, inception_3a_pool_proj_bn_out], 1)
  
    inc = self.inception_3b_1x1(inception_3a_output_o)
    inception_3b_1x1_b_n_out = self.inception_3b_1x1_bn(inc)
   
  
    inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_b_n_out)
    inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_o)
    INCEPTION_3B_3X3_REDUCE_BN_OUT = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
   
  
   
    INCEPTION_3B_RELU_3X3_REDUCE_OUT = self.inception_3b_relu_3x3_reduce(INCEPTION_3B_3X3_REDUCE_BN_OUT)
    inception_3b_3x3_outRk = self.inception_3b_3x3(INCEPTION_3B_3X3_REDUCE_BN_OUT)
   #hmUJXyPt
    INCEPTION_3B_3X3_BN_OUT = self.inception_3b_3x3_bn(inception_3b_3x3_outRk)
    inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(INCEPTION_3B_3X3_BN_OUT)
    inception_3 = self.inception_3b_double_3x3_reduce(inception_3a_output_o)
    inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(inception_3)
    inception_3b_relu_double_3x3__reduce_out = self.inception_3b_relu_double_3x3_reduce(inception_3b_double_3x3_reduce_bn_out)

    inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_double_3x3_reduce_bn_out)
    inception_3b_double_3x3_1_bn_outJujO = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
    inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_outJujO)
    inception_3b_do = self.inception_3b_double_3x3_2(inception_3b_double_3x3_1_bn_outJujO)
 
   
   
    inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_do)
    inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
    inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_o)
    inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
 
    inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
    inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
    inception_3b_output_out = torch.cat([inception_3b_1x1_b_n_out, INCEPTION_3B_3X3_BN_OUT, inception_3b_double_3x3_2_bn_out, inception_3b_pool_proj_bn_out], 1)
    inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(inception_3b_output_out)
  
 #vqVekfdHlWygjhrau
    inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
    inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
    inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_3x3_reduce_bn_out)
 
    inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
    inception_3c_rel_u_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
    inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(inception_3b_output_out)
    inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(inception_3c_double_3x3_reduce_out)
    inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(inception_3c_double_3x3_reduce_bn_out)
  
    inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_double_3x3_reduce_bn_out)
   
    inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
    inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
    incepk = self.inception_3c_double_3x3_2(inception_3c_double_3x3_1_bn_out)
   
   
    inception_3c_double_3_x3_2_bn_out = self.inception_3c_double_3x3_2_bn(incepk)
    incepj = self.inception_3c_relu_double_3x3_2(inception_3c_double_3_x3_2_bn_out)
    inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
    inception_3_c_output_out = torch.cat([inception_3c_3x3_bn_out, inception_3c_double_3_x3_2_bn_out, inception_3c_pool_out], 1)

    inception_4a_ = self.inception_4a_1x1(inception_3_c_output_out)#NWUapqtwrgo
    inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_)
    incep_tion_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
    incepti = self.inception_4a_3x3_reduce(inception_3_c_output_out)
    inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(incepti)

    inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
    inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_3x3_reduce_bn_out)
    inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
    inception__4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
    inception_4a_double_3x3_reduce_ou = self.inception_4a_double_3x3_reduce(inception_3_c_output_out)
    inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(inception_4a_double_3x3_reduce_ou)
    inception_4a_rel_u_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(inception_4a_double_3x3_reduce_bn_out)
    inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_double_3x3_reduce_bn_out)
   
    inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
  
 
  #CtZNkBhaeuA
    inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
    inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_double_3x3_1_bn_out)
   
    inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
    inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
  #jGVYIsbiBuQwptqeC
   #WuUf
   
    inception_4a_pool_out = self.inception_4a_pool(inception_3_c_output_out)
    inception_4a_pool_proj_o = self.inception_4a_pool_proj(inception_4a_pool_out)
   

   
    inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_o)
    inceptior = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
    inception_4a_output_out = torch.cat([inception_4a_1x1_bn_out, inception_4a_3x3_bn_out, inception_4a_double_3x3_2_bn_out, inception_4a_pool_proj_bn_out], 1)
    inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
   
    inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
    inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
 #CpySOszBhkRqjPuLZ

    inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(inception_4a_output_out)
 
    INCEPTION_4B_3X3_REDUCE_BN_OUT = self.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
   
 
    inception_4b_ = self.inception_4b_relu_3x3_reduce(INCEPTION_4B_3X3_REDUCE_BN_OUT)
    inception_4b_3x3_out = self.inception_4b_3x3(INCEPTION_4B_3X3_REDUCE_BN_OUT)
    inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
  
    inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
    inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(inception_4a_output_out)
    inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(inception_4b_double_3x3_reduce_out)#uCngxtePOGozVFsipT
    inceptio_n_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(inception_4b_double_3x3_reduce_bn_out)
    inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_double_3x3_reduce_bn_out)
    inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
    inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
   #yc
    incep = self.inception_4b_double_3x3_2(inception_4b_double_3x3_1_bn_out)
    inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(incep)
    inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
    inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
  
    inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
    inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
    inception_4b_r_elu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
    inception_4b_output_out = torch.cat([inception_4b_1x1_bn_out, inception_4b_3x3_bn_out, inception_4b_double_3x3_2_bn_out, inception_4b_pool_proj_bn_out], 1)

    inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
#JFVDovHn
    inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
    inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
    inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(inception_4b_output_out)
    inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
    inception_4c_relu_3x = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
    inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_3x3_reduce_bn_out)
    INCEPTION_4C_3X3_BN_OUT = self.inception_4c_3x3_bn(inception_4c_3x3_out)
    inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(INCEPTION_4C_3X3_BN_OUT)
    INCEPTION_4C_DOUBLE_3X3_REDUCE_OUT = self.inception_4c_double_3x3_reduce(inception_4b_output_out)
    inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(INCEPTION_4C_DOUBLE_3X3_REDUCE_OUT)
    inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(inception_4c_double_3x3_reduce_bn_out)
    inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_double_3x3_reduce_bn_out)
    inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
    inceptio_n_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
 
  
    inception_4c_double_3x3_2_outAyCmL = self.inception_4c_double_3x3_2(inception_4c_double_3x3_1_bn_out)
    inception_4c_double_3x3_2 = self.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_outAyCmL)
  
  
    incept_ion_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2)
   
  
   
  
    inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
   
    inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
    inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
    inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
    inception_4c_output_out = torch.cat([inception_4c_1x1_bn_out, INCEPTION_4C_3X3_BN_OUT, inception_4c_double_3x3_2, inception_4c_pool_proj_bn_out], 1)
   #nrFDIvXJpqtifLPN
    inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
  
    inception_4d_1x1_bn_outOft = self.inception_4d_1x1_bn(inception_4d_1x1_out)
    inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_outOft)
#rjnlYmSPuyMBXgW
    inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(inception_4c_output_out)

  
 
    inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
    inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
 
    inception_4d_ = self.inception_4d_3x3(inception_4d_3x3_reduce_bn_out)

   
    inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_)
  
    inception_4d_r = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)#PWfFpkgbHd
    inception_4d_double_3x3_reduce_outhUrJ = self.inception_4d_double_3x3_reduce(inception_4c_output_out)
  
   
    inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(inception_4d_double_3x3_reduce_outhUrJ)
    inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(inception_4d_double_3x3_reduce_bn_out)
    inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_double_3x3_reduce_bn_out)
    inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
 
    inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
    inception_4d_doub = self.inception_4d_double_3x3_2(inception_4d_double_3x3_1_bn_out)
   
    inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(inception_4d_doub)
    inception_4d_relu_doubl = self.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
   
    inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
    inception_4d_pool_proj = self.inception_4d_pool_proj(inception_4d_pool_out)
   
 
    inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj)
 
    inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
    inception_4d_output_out = torch.cat([inception_4d_1x1_bn_outOft, inception_4d_3x3_bn_out, inception_4d_double_3x3_2_bn_out, inception_4d_pool_proj_bn_out], 1)

    incepX = self.inception_4e_3x3_reduce(inception_4d_output_out)#YwqVcaWZbsIrXFKBp
  
    inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(incepX)
    inception_4e_relu__3x3_reduce_out = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
  
    inception_4e_ = self.inception_4e_3x3(inception_4e_3x3_reduce_bn_out)
    inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_)#EpCDJQ
    inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
    inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(inception_4d_output_out)#wafRhmMo
  #GsRIY
    inception = self.inception_4e_double_3x3_reduce_bn(inception_4e_double_3x3_reduce_out)
    inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(inception)
   
 
    inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(inception)
    inception_4e_double_3x3__1_bn_out = self.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
    inception_4e_rel = self.inception_4e_relu_double_3x3_1(inception_4e_double_3x3__1_bn_out)
    inception_4e_double_3x3_2_o = self.inception_4e_double_3x3_2(inception_4e_double_3x3__1_bn_out)
    inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_o)


    inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
  
    inception_4e_pool_outl = self.inception_4e_pool(inception_4d_output_out)
    inception_4e_output_out = torch.cat([inception_4e_3x3_bn_out, inception_4e_double_3x3_2_bn_out, inception_4e_pool_outl], 1)
    inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
   
    inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
    inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
    inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(inception_4e_output_out)
    inception_5a_3x3_reduce_bn_outsChS = self.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
    inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_outsChS)
    inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_3x3_reduce_bn_outsChS)
    inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
    incept_ion_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
    incepti_on_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(inception_4e_output_out)
   
    inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(incepti_on_5a_double_3x3_reduce_out)
    INCEPTION_5A_RELU_DOUBLE_3X3_REDUCE_OUT = self.inception_5a_relu_double_3x3_reduce(inception_5a_double_3x3_reduce_bn_out)
    inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_double_3x3_reduce_bn_out)
    inception_5a_double_3x3_1_bn = self.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
   

    inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn)
    inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_double_3x3_1_bn)
   
    inception_5a_d_ouble_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
  
   
    inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(inception_5a_d_ouble_3x3_2_bn_out)
    inception_5a_pool_o = self.inception_5a_pool(inception_4e_output_out)
    inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_o)
   
    inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
    inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
    inception_5a_output_out = torch.cat([inception_5a_1x1_bn_out, inception_5a_3x3_bn_out, inception_5a_d_ouble_3x3_2_bn_out, inception_5a_pool_proj_bn_out], 1)
    inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
    inc_eption_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
   
  #spGMISEAKVwTm
   
   
    inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inc_eption_5b_1x1_bn_out)
    INCEPTION_5B_3X3_REDUCE_OUT = self.inception_5b_3x3_reduce(inception_5a_output_out)
  
    inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(INCEPTION_5B_3X3_REDUCE_OUT)

    inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
   
    INCEPTION_5B_3X3_OUT = self.inception_5b_3x3(inception_5b_3x3_reduce_bn_out)
  
    inception_5b_3x3_bn_outcbSiq = self.inception_5b_3x3_bn(INCEPTION_5B_3X3_OUT)
    incepp = self.inception_5b_relu_3x3(inception_5b_3x3_bn_outcbSiq)
    incep_tion_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(inception_5a_output_out)
    inception_5b__double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(incep_tion_5b_double_3x3_reduce_out)
    inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(inception_5b__double_3x3_reduce_bn_out)
    inception_5b_ = self.inception_5b_double_3x3_1(inception_5b__double_3x3_reduce_bn_out)
    inceptio = self.inception_5b_double_3x3_1_bn(inception_5b_)
    inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(inceptio)#j
   
    inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(inceptio)
   #tMiczbSysWhjXrxlU
 
    inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)#FeDERXpCuZmaTwvsyHrW
    inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
    inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)

    inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
    inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
    inception_5b_rel_u_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
    inception_5b_output_out = torch.cat([inc_eption_5b_1x1_bn_out, inception_5b_3x3_bn_outcbSiq, inception_5b_double_3x3_2_bn_out, inception_5b_pool_proj_bn_out], 1)
  
    return inception_5b_output_out
   

  def forward(self, input):
    _x = self.features(input)
   #pdJIlwx
 #GLXYu
    return _x#rBOuUDIcA

  def __init__(self, embedding_size, pretrainedIn=True, i=True):
  
   
  
    """  Ǜ         """
  
   
   

    super(BNInception, self).__init__()
  

    inplace = True
  
    self.embedding_size = embedding_size
    self.num_ftrs = 1024
    self.is_norm = i#PkEvoqGVXmpQ
    self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
    self.conv1_relu_7x7 = nn.ReLU(inplace)
    self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
    self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
    self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)

    self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
  
    self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv2_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
    self.conv2_relu_3x3 = nn.ReLU(inplace)
   
    self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
  
    self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
   
 
    self.inception_3a_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)#IjNK
    self.inception_3a_relu_1x1 = nn.ReLU(inplace)
    self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
    self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
    self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
    self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_3a_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
    self.inception_3a_relu_3x3 = nn.ReLU(inplace)
    self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
 
 
  
    self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
 
  
   
   
   
    self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
  
    self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 
    self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)#Y
  
    self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
    self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
  
   
#sEvJNurachWjVRoYfkD
    self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
    self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
   
   
  
    self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
    self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True)
    self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
   
    self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
   
   
    self.inception_3b_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
    self.inception_3b_relu_1x1 = nn.ReLU(inplace)
    self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
    self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
    self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_3b_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
    self.inception_3b_relu_3x3 = nn.ReLU(inplace)
    self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)

    self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
    self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
    self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
  
   
    self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)

    self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)#Vx
    self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
  
  
    self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
    self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
    self.inception_3b_relu_pool_proj = nn.ReLU(inplace)

  
    self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1))
    self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)#DzBcR

    self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
  

  
    self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.inception_3c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
    self.inception_3c_relu_3x3 = nn.ReLU(inplace)
    self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
    self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
 
    self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
   
   
    self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
   

    self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
    self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
 
   
  
    self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
    self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
    self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1), stride=(1, 1))
   
    self.inception_4a_1x1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4a_relu_1x1 = nn.ReLU(inplace)
    self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
    self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_4a_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
  
    self.inception_4a_relu_3x3 = nn.ReLU(inplace)
    self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)

    self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
   
    self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
    self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
    self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
 
  
 
    self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
 
    self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
   
    self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
  
    self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1))
  
  
    self.inception_4b_1x1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4b_relu_1x1 = nn.ReLU(inplace)
    self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
   
    self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)#XmdWgwOD
    self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
    self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_4b_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)

  
    self.inception_4b_relu_3x3 = nn.ReLU(inplace)
    self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
  

    self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)#ZWuSgMhfikTCotRIe
    self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
    self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
    self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
    self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
    self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4c_1x1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4c_relu_1x1 = nn.ReLU(inplace)
   
    self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
 
    self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
    self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   
    self.inception_4c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4c_relu_3x3 = nn.ReLU(inplace)
    self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
    self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
    self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 
    self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
    self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
    self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
  
   

  
    self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)

    self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
    self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4d_1x1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
   #YlpUsEK
    self.inception_4d_relu_1x1 = nn.ReLU(inplace)
    self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
    self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_4d_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4d_relu_3x3 = nn.ReLU(inplace)
    self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
    self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


 
    self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
    self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)#OyWEAUrdaSYoKZHnXJ
    self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
    self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
    self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
    self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
    self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.inception_4e_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4e_relu_3x3 = nn.ReLU(inplace)
    self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192, kernel_size=(1, 1), stride=(1, 1))
    self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
  
    self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   
    self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
  
  
   
    self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
  
    self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
    self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
    self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
    self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1), stride=(1, 1))

    self.inception_5a_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)#vlEx
  
    self.inception_5a_relu_1x1 = nn.ReLU(inplace)
    self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1))
    self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
  
    self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
    self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    self.inception_5a_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
    self.inception_5a_relu_3x3 = nn.ReLU(inplace)
   
 
    self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160, kernel_size=(1, 1), stride=(1, 1))

    self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)

    self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
 
    self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
    self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
  #c
    self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
   
    self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
    self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
  
    self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 1), stride=(1, 1))
 
    self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
    self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1), stride=(1, 1))
    self.inception_5b_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)
    self.inception_5b_relu_1x1 = nn.ReLU(inplace)
    self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
    self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
 
  
    self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)

    self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_5b_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
    self.inception_5b_relu_3x3 = nn.ReLU(inplace)
    self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
    self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
   
    self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
   
    self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))#vAu
    self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
    self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
    self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)#PlJT
    self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
    self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=True)
    self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
    self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
    self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
    self.global_pool = nn.AvgPool2d(7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
#XbQcfz
    self.last_linear = nn.Linear(1024, 1000)

   
  def l2_normtQ(self, input):
    inp_ut_size = input.size()
    buffer = torch.pow(input, 2)
  
    normpOYVuJ = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normpOYVuJ)
  
    _outputMON = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _outputMON.view(inp_ut_size)
    return output
   

