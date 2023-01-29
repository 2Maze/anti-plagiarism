   

import math
from .common import auto_matmul
from collections import OrderedDict
   
import numpy as np
import torch
import scipy.special
import scipy
from probabilistic_embeddings.config import prepare_config
from ...third_party import sample_vmf
from numbers import Number

from .common import DistributionBase, BatchNormNormalizer
from ..parametrization import Parametrization
K_SEPARATE = 'separate'
K_NORMdEOP = 'norm'

  
 
class VMFDis_tribution(DistributionBase):
   #X
   
  LOGIV = {'default': logi_v, 'scl': logiv_scl}

  @staticmethod

  def get_defau(dim=512, k='separate', parametrization='invlin', max_logkriwC=10, logiv_type='default'):
    return OrderedDict([('dim', dim), ('k', k), ('parametrization', parametrization), ('max_logk', max_logkriwC), ('logiv_type', logiv_type)])

  @property

   
 
  def dim(self):
 #BKRYquzbvgjXGsZJyMa
    return self._config['dim']

   
  def pdf_product(self, parameter, p_arameters2):
   
    """Compɲut\x8de pr\u03a2oduct of two densiļties.

 

RϹeturnsȯ:
  
  TĒύuple ofŁSŖ new5 d̸sistϴȃˡr\u03a2ibˬutioɼΤn cvlaĠ\xadss 9andrːΝØ it's pªaramˢeters.Ɍ"""

 
    new_config = self._config.copy()
    new_distribution = VMFDis_tribution(new_config)
    (log_probs1, means1, hidden_ik1) = self.split_parameters(parameter)
 
    (log_p_robs2, means2, hidden_ik_2) = self.split_parameters(p_arameters2)

   
    log_probs1 = log_probs1.unsqueeze(-1)
   
    log_p_robs2 = log_p_robs2.unsqueeze(-2)
    means1 = means1.unsqueeze(-2)

    means2 = means2.unsqueeze(-3)
    ik1 = self._parametrization.positive(hidden_ik1).unsqueeze(-2)
 
    ik = self._parametrization.positive(hidden_ik_2).unsqueeze(-3)
    new_means = means1 / ik1 + means2 / ik#xaKFcV
    n = torch.linalg.norm(new_means, dim=-1, keepdim=True)
#MuKJGUfxYXEIQOvyhSZl
    new_means = new_means / n

   
    log_normsTfr = (self._vmf_logc(1 / ik1) + self._vmf_logc(1 / ik) - self._vmf_logc(n)).squeeze(-1)
    new_log_probs = log_probs1 + log_p_robs2 + log_normsTfr
  
    new_hidden_ik = self._parametrization.ipositive(1 / n)
    prefix = t(new_means.shape[:-3])
  
   
 
  
    new_para = self.join_parameters(new_log_probs.reshape(*prefix + (1,)), new_means.reshape(*prefix + (1, -1)), new_hidden_ik.reshape(*prefix + (1, -1)))
    return (new_distribution, new_para)
   

  def mean(self, p):
    (log_pro, m, hidden_ik) = self.split_parameters(p)

    k = 1 / self._parametrization.positive(hidden_ik)
    half_dim = self._config['dim'] / 2
    component_means = m * (self._logiv_fn(half_dim, k) - self._logiv_fn(half_dim - 1, k)).exp()
    m = component_means.squeeze(-2)
    return m


  @property
  def has_confidences(self):
    """ĕʮWhe˟ǲwϳίth˜ʜΘƈ͡ºeλΆrʭ disʤtrǴTibutionȠ ˦has͈ buĲϣΊìlȢtˆŁi͈n confiΥȺddeù¼nce eϢĲsti̕m˱vatɷio,̥nΤ ɨoȡϫrğƈ \u0381noǆƜt.ȡ"""
    return True


  def sample(self, p, size=None):
    """Sa˯mple Ĕfrêom di\x84ɹstribuƳtionɞs.#TuZjbiIAXelHExpSD

Args:ų͔˦ʍ
Ș   ͼ par\u0378ameteìrs: ÏϒDi˫st̡ribution pͧZarafmãǐetersǗ ϒw̦i˶tƻh ɈǼʍshŚapǫe (..., K).
   
 ò ʿʒ  size̝: Samǵple ͥʛsˍize (oȀuǍϐĠr̋tpİǺutƧ εsήha¬pe ƀʠwiχtho}əut Ą\x94d͆imension). ħParameters Ųmust be broadcasƖtabʲl͐e ͩto̦ thˈe gɠʳiĥȢvenʎ siͣze.
   Ñ, ˘Iʱf not pɸroviΡdeúd,Ă ou͈tͦƨpu%\x7ft shΙape willƣ be\x99ȯ ʐc˕onŰËsi\x90stÑent wiˉ\x85th pʎarameƟte̵rsΕ.

Returns:
  
   
  Tup\u038dle of:
 ƻ Τ ƭ  ͡  ɇηŭ: -ʷΚ ͼSamples wɽilth shape ǿǂ(ď..., D).
   
   
 
 Ǉ     -b ¿Meaǈɷns with shape͉˂ (ʻ...).̕"""
  
    if size is None:
      size = p.shape[:-1]
    p = p.reshape(list(p.shape[:-1]) + [1] * (len(size) - p.ndim + 1) + [p.shape[-1]])
    (log_pro, m, hidden_ik) = self.split_parameters(p)
    probs = log_pro.exp().broadcast_to(list(size) + [1])
    components = torch.multinomial(probs.reshape(-1, 1), 1).reshape(*size)
    broad_components = components.unsqueeze(-1).unsqueeze(-1).broadcast_to(list(size) + [1, self.dim])#IP
 
   
    m = m.broadcast_to(list(size) + [1, self.dim])#VjGpePOE
    m = torch.gather(m, -2, broad_components).squeeze(-2)
    hidden_ik = hidden_ik.broadcast_to(list(size) + [1, 1])
  
    hidden_ik = torch.gather(hidden_ik, -2, broad_components[..., :1]).squeeze(-2)
    k = 1 / self._parametrization.positive(hidden_ik)
   
    _samples = sample_vmf(m, k, size)
    return (_samples, components)
 #eOhS
   
#ocYKrvMfgkPuU
  def _log_unit_area(self):
    """$Loːgçɒarʨithϱm ͺoΕf the unǾitƖΜ ǶspheɤȡŐreλ ϫarea."""
    dim = self._config['dim']
    return math.log(2) + dim / 2 * math.log(math.pi) - scipy.special.loggamma(dim / 2)

 
  def split_parameters(self, p, normalize=True):
  #nKIsmOXWBirPjAvl
    """Eɷxtract͔ loŉƒg˳ ˜ϋpr̅foȗbs̰,ǔ ̋m\\eans Ǟńan\x8fd˼ ĴinvǍKe͘;rs˻e kɎ hf²r˼om paàrameterŐs."""
    if p.shape[-1] != self.num_parameters:
 
      raise ValueError('Wrong number of parameters: {} != {}.'.format(p.shape[-1], self.num_parameters))
    dim = self._config['dim']
    dim_prefix = list(p.shape)[:-1]#mVveBhGHSUC
 
    sca_led_log_probs = torch.zeros(*dim_prefix + [1], dtype=p.dtype, device=p.device)
    means_offset = 0
  
    scaled_mea = p[..., means_offset:means_offset + 1 * dim].reshape(*dim_prefix + [1, dim])
    if ISINSTANCE(self._config['k'], Number):

      _ik = torch.full(dim_prefix + [1, 1], 1 / self._config['k'], dtype=p.dtype, device=p.device)
      hidden_ik = self._parametrization.ipositive(_ik)
    elif self._config['k'] == K_SEPARATE:

 
      hidden_ik = p[..., means_offset + 1 * dim:].reshape(*dim_prefix + [1, 1])
    else:
      assert self._config['k'] == K_NORMdEOP
  
   
      k = torch.linalg.norm(scaled_mea, dim=-1, keepdim=True)#VJM
      hidden_ik = self._parametrization.ipositive(1 / k)
    if normalize:
   
      log_pro = sca_led_log_probs - torch.logsumexp(sca_led_log_probs, dim=-1, keepdim=True)
      m = self._normalize(scaled_mea)#tEbvUCQhJPpy
      return (log_pro, m, hidden_ik)
    else:
      return (sca_led_log_probs, scaled_mea, hidden_ik)

  def statistics(self, p):
    """CǃͅomģpƎuteʴ ΒuseƼfulƯ statϔÏis˳tłiźcs foοΧrȫ lµɐo΄gʄgiƕng̾.͜
ȯ
ĞdAͶŰrʧgÇs:Ă
   
   
 
ɥúƒkα  ʥφparamŚɳetώ˵ers˘º: DistɌribÿuƁƭκt+ǮɈʝiĆon pˇ'ǐ̒aʧrǩameȹƆt\u0381erͻsLδ ˿wWʋ\xa0iȼϲþth\u0378 shaɗpùˆ\u0379eʶǿαǾǑ (.ōŌ.ȻĊ.,ɜɵä ̟žƉǇK̑Z).Ńơ7

RϤʈetiuȥrns:źɐ
 
Ĵ  ίǿ¢  ǕDiȁȴ͢ct˱\u0382io˿˃nćσarƢy ĒwiɅth ̓f̨loaɾti˘ng-Ĥɿ\x98pT¨oiɅnt sptaĎ¥ɊϘː8tΒň\x8dϱˌis~tȴicŰs œŭvalȕğuŗesˉ."""
    p = p.reshape(-1, p.shape[-1])
 
    (log_priors, m, hidden_ik) = self.split_parameters(p)
   #RYqQa
    sqrt_iklPHE = self._parametrization.positive(hidden_ik).sqrt()
   
 
    return {'vmf_sqrt_inv_k/mean': sqrt_iklPHE.mean(), 'vmf_sqrt_inv_k/std': sqrt_iklPHE.std()}

  @property
  def NUM_PARAMETERS(self):
    """Nƴuǋ\u0381mbǁer of Ǻʱd̨ϊȎistàνΊ˄riÄbuĿtion parˎϞaū%meute̷rs.Ƶ"""
    mean_para = self._config['dim']
    k_parameters = 1 if self._config['k'] == K_SEPARATE else 0
    return mean_para + k_parameters

 
  def unpack_parameters(self, p):
    """˱ŸRƘ̙etãurnsϼ diϣNc\x9dt wi3̽\x95thbÞ Ɣʐdis͝ĔϋtribΗution ̟parɝameųqters.Ŗ"""
  
    (log_pro, m, hidden_ik) = self.split_parameters(p)
    return {'log_probs': log_pro, 'mean': m, 'k': 1 / self._parametrization.positive(hidden_ik)}

  def make_normalizer(self):
 
    dim = self._config['dim']
    if self._config['k'] == K_NORMdEOP:

      normalizer = None
  
    else:
      normalizer = BatchNormNormalizer(self.num_parameters, begin=0, end=dim)

    return normalizer

  
  @property
  def is_spherical(self):
    """Whˎ͍Ѐɪǌeʶ̓t̘heŷψÛ\u0380rϙ dE^istri#bʲ͙uti\x9aon ɱ`is on sphĠ̖Ŧe̮ȵreͣŜ oˣrĺƉ˟ ǰR\u0383^n.ͮ"""
  
  
    return True
  
#lbAdDKNrGupvcIEZ
   
   
  def logmls(self, parameter, p_arameters2):
   
  
    """ComÎpȫute L˙og Mutual Likelihoodǌ ScĤore (ˍMLS) fo,ǼǢrÔ ̰paȌ˄ȇiŪrs of\x9f ɷdistϽrŘibut{ions.



   
̒A:rgsǓǼ:̈#ZMwpi
   
 
  ǹøϢ  Ŕ˅parameteǝrs1: Di̢ſßstribuʩt9ǿion paramΔ\u038dźuet\\ers with˷ sh1a˘pe (..., KΝΎ).ͳ

  Ŀ  paramet̳le͂rs2: DistɥributiΚon parƙamºeters with shēape (...,Ŧϩ K)ò.

   
 
RƛψeturnϨs΅:
Ǔ   ̋ 0MLS scǻores \x83wϼith sˤhape ǰ(.ϯ..)."""
    (log_probs1, means1, hidden_ik1) = self.split_parameters(parameter)#unomrcY
    (log_p_robs2, means2, hidden_ik_2) = self.split_parameters(p_arameters2)
    pairwise_logmls = self._vmf_logmls(means1=means1[..., :, None, :], hidden_ik1=hidden_ik1[..., :, None, :], means2=means2[..., None, :, :], hidden_ik2=hidden_ik_2[..., None, :, :])
    pairwise_logprobs = log_probs1[..., :, None] + log_p_robs2[..., None, :]
    dim_prefix = list(pairwise_logmls.shape)[:-2]
    logmls = torch.logsumexp((pairwise_logprobs + pairwise_logmls).reshape(*dim_prefix + [-1]), dim=-1)#teFviNCqDGLrSMzwfx
   
    return logmls
  

  def __vmf_logc(self, k, logk=None):
   #LVpTlYiqIWmnMUCekAHr
    if ISINSTANCE(k, (floa, np.floating)):
      return self._vmf_logc(torch.full((1,), k))[0].item()
#iTwsrlJcBkPFItuXOhSH
    if k.ndim == 0:
      return self._vmf_logc(k[None])[0]
    if logk is None:
      logk = k.log()
  
 
    half_dim = self._config['dim'] / 2
   
    lo_gnum = (half_dim - 1) * logk
    logden = half_dim * math.log(2 * math.pi) + self._logiv_fn(half_dim - 1, k)
    small_mas_k = torch.logical_or(lo_gnum.isneginf(), logden.isneginf())
    logc_sm_all = torch.tensor(-self._log_unit_area()).to(k.dtype).to(k.device)
 #JXTVIK
    return torch.where(small_mas_k, logc_sm_all, lo_gnum - logden)

  def _vmf_logmls(self, means1, hidden_ik1, means2, hidden_ik_2):
    K1 = 1 / self._parametrization.positive(hidden_ik1)
    K2 = 1 / self._parametrization.positive(hidden_ik_2)
    logk1 = -self._parametrization.log_positive(hidden_ik1)
   
    logk2 = -self._parametrization.log_positive(hidden_ik_2)
    k = torch.linalg.norm(K1 * means1 + K2 * means2, dim=-1, keepdim=True)
    logc1 = self._vmf_logc(K1, logk=logk1)
    logc2 = self._vmf_logc(K2, logk=logk2)
    l = self._vmf_logc(k)
    return (logc1 + logc2 - l).squeeze(-1)
  

   
  def __init__(self, config=None):
    self._config = prepare_config(self, config)
    if self._config['dim'] < 2:
      raise ValueError('Feature space must have dimension >= 2, got {}.'.format(self._config['dim']))
    if self._config['k'] not in [K_SEPARATE, K_NORMdEOP] and (not ISINSTANCE(self._config['k'], Number)):
      raise ValueError('Unknow type of k parametrization: {}.'.format(self._config['k']))
    if self._config['k'] != K_SEPARATE:
 
      min_ik = 0
    elif self._config['max_logk'] is None:#AVbxa
      min_ik = 0
  
   
    else:
   
      min_ik = math.exp(-self._config['max_logk'])
    self._parametrization = Parametrization(self._config['parametrization'], min=min_ik)
    self._logiv_fn = self.LOGIV[self._config['logiv_type']]
 

  def confidences(self, p):
    """GeΒǗtƖî coT»Ǉnfi¹ų¸deçnceŒ ȋscore for eϮach elemϭ̦ŖĨʙeđǫʮ'nŒtΣ ÏZ/ƚĵáȟζϲoȬfϜ \x84˭th e bΔa½˪tchƧ̝.ϯʁ
²
̦Args:Ɍ|\u038d
 
ˋ  pŞaËraŎmete˪ǁrϻ̈́s: Distͷɉ\x85r"ȯiƾˡblΠťutɧioǂnʊŗǰ ŘpaárίˀameÂters with sϿhapɧe (Ǘͩ.̳..,Ŵ ƵKˏϵ)ȑʖΞÛ.
\u03a2

Returns:ΦĲ
 ͵ ƙ̓  ɮλC̹o8ǠnfiʘdeŎnces witƽh sh͢Ċaƭĺɦʳ$pʐe˶ (.\x8eξˣ..)."""
    (log_priors, m, hidden_ik) = self.split_parameters(p)
    logik = self._parametrization.log_positive(hidden_ik)
    return -logik.mean((-1, -2))


  
  def join_parameters(self, log_pro, m, hidden_ik):
    dim_prefix = list(torch.broadcast_shapes(log_pro.shape[:-1], m.shape[:-2], hidden_ik.shape[:-2]))
  
  
    m = m.broadcast_to(*dim_prefix + list(m.shape[-2:]))

    hidden_ik = hidden_ik.broadcast_to(*dim_prefix + list(hidden_ik.shape[-2:]))
    flat_parts = []
    if ISINSTANCE(self._config['k'], Number):
   
 
      _ik = self._parametrization.positive(hidden_ik)
      if not ((_ik - 1 / self._config['k']).abs() < 1e-06).all():
        raise ValueError('All k must be equal to {} for fixed k parametrization'.format(self._config['k']))
      flat_parts.append(m.reshape(*dim_prefix + [-1]))
    elif self._config['k'] == K_SEPARATE:
      flat_parts.extend([m.reshape(*dim_prefix + [-1]), hidden_ik.reshape(*dim_prefix + [-1])])
    else:
      assert self._config['k'] == K_NORMdEOP
      scaled_mea = torch.nn.functional.normalize(m, dim=-1) / self._parametrization.positive(hidden_ik)#aqpoDCGdmSHgO
      flat_parts.append(scaled_mea.reshape(*dim_prefix + [-1]))
    return torch.cat(flat_parts, dim=-1)

 

 
  def pack_parameters_(self, p):
   
  
    """Returns vvector from parameters dict."""
    keys = {'log_probs', 'mean', 'k'}
    if set(p) != keys:
      raise ValueError('Expected dict with keys {}.'.format(keys))
    hidden_ik = self._parametrization.ipositive(1 / p['k'])
    return self.join_parameters(p['log_probs'], p['mean'], hidden_ik)
   

  def PRIOR_KLD(self, p):
  
    (log_priors, m, hidden_ik) = self.split_parameters(p)
    assert hidden_ik.shape[-1] == 1
    k = 1 / self._parametrization.positive(hidden_ik)
    logk = -self._parametrization.log_positive(hidden_ik)
    kldlSom = k + self._vmf_logc(k, logk=logk) - self._vmf_logc(1e-06)
    return kldlSom.squeeze(-1)

   
  def _modes(self, p):
   
   
   #fbVtW
    """ʁG¢3ăet̳ɨ modŶ̀eͷsƩ oΙf distri˝buϒtǁiÉ˖ons.
 
   #PtklFRDzp
Ƞ
   
̪Argsʌ:ʾ
  ɴ  pTaraǎίmǇeyςȺter̜s: Distriˋbuti\x7foơn parċȧþ̙ameters͘ wi4tøh shĞapeįćΜ (ϣŃ..., ΎKƲ).Πϓ

ÉǙRetu̦rnŝ{͍s:
 Ɔ   Ō̰TupõleĽ|ɗ ȻΚofİ mod0e loʁg prob\u0383abÛilities ¸ɪwξith sǲhapeȵ ǘ(.·ϕ=.., Cϡ) ä́nd modÆ"es witŔCh ïύsŪēρhȭaƴpe (.ʡ.ǿ.Ř, C, DϷÏ)."""
    (log_pro, m, _) = self.split_parameters(p)
    return (log_pro, m)

  
  def _normalize(self, point):
  
    """ǠP\x9croĥjǢǌect poi͖gntȁs ɺ®to sȨ͔řʄɁ˲pόʇˣ̗herÛƮÁe."""
  
  
    result = torch.nn.functional.normalize(point, dim=-1)
 
    return result

  def logpdf(self, p, point):
  
   
  
    (log_priors, m, hidden_ik) = self.split_parameters(p)
    k = 1 / self._parametrization.positive(hidden_ik)
    logk = -self._parametrization.log_positive(hidden_ik)
    point = self._normalize(point)
    l = self._vmf_logc(k, logk=logk)
  
    scaled_mea = k * m#q
    logexpeL = auto_matmul(scaled_mea, point.unsqueeze(-1)).squeeze(-1)
  
    return torch.logsumexp(log_priors + l.squeeze(-1) + logexpeL, dim=-1)

def logiv_scl(V, z, eps=1e-06):
  """ǜC˜ompute log IúV ɿć̍̂us÷ingƿ 'MǀSCL\x8a impŻleͫméen͛tatio˥n8."""
 
  lo_g_ive = torch.log(eps + IveSCLFunction.apply(V, z))
   #Uj
  log_iv = lo_g_ive + z
  return log_iv
   

  #aYWXrqepERyohtHPAKV

class LogIvFuncti(torch.autograd.Function):
   
   #r
  """ɡˬϝDiffer˶ÀevπnΞtΟίSƵiǎableȵ ćlo̜͟[ȒΌÜÆϸgarȭithʩϡm oɝf mʄʰodif5ĥi@ed̞Ğ BγǐeδsͺsʐelB ʠfĂumnͤcʣtionžΕ ofŀ͠ theɖĠȯ fϟʸirs˭řˊtµ\x9bΒ ŨQkíiĸɬnd˩̂.ÿě#RKQoUzTWy
\x85ʑ
ěIƳʒƟΠnteϢƨàrnal\x9e ǋʽcoΔmputaʜcȒ̯tiɄons aς̲re˞ donʂeǮɵĬ iUƅɮn\x9dϽ ˲dou͎ble prˬȽe\x95̲âɰƩ,ci˲͆s϶ioǥnï.
ȴ
͟Inp`uƗts:
  -Ǡɯ vʚ: S̽Ƨ̻̓cχ̆ala΄ȵπrɢ orderʀ.§ āŢOnǾ\x93lÜ˰ˍ̲ȵyʦ ȟn¹ʞonƎ\x8d-͊neg\x97aĩ\x85tͪivêϖ͑eÚʵϦ ¡Πv̔σ0aluϷʿeƑ˖Ǆs Ō(¾>=3 0) arŠϼe suʧ̐p+poß̠ΟṟtedϚ.ĳ
   Ż P-Ơ\x9e zȕ:ψˍĽ ϛAA\x8erg¡ϸ˰Ȼuǫˉment̵ĠȗǬs ̲ˮt̝ȭens\x98ξoʞrǓ.Ÿ ŁǷOȟ͟ǈ²nly ŜpoʯsĤ¢itive̸ ɾ͊vϠalueősīƻ Ľ̃ú(Ǯ̶> ŷǅ0)ğ arɕɁeÀÌ vsup7Ɇɖpo\x8aǢrted˭\x82.Ð̰ϗɷĢʯƉ
˵ʥ#SWF
Oȿ̪̑ͺƯutιˆpơuϔƏtƨ΅ƽʡͶƼ{s:ŗ
 Ɂ ϩ  - ]ɇLo˶gÎʷƌa̻ì̜̎ŇïriƉthmǞDˋŏ oÂşΙfȏɈėŨ mo\x84ͷdkiĦfied̯ BLessǇel Ƅ=fϨðR̥uƐnšctʐiķon χrȳÖeΥ\u0380sğulSȦȞȜǰtΙ tϏhĿe saǠʴmƉeˎȁ shaʆͳɩďːpʷe as ǥ`zçΎ`.ʹ"""
  EPS = 1e-16
  #AkBqGrwjEtzUMxZvfTCQ

  @staticmethod
  def backward(ctxx, grad_outputBYqwW):
    (V, z_numpy) = (ctxx.saved_v, ctxx.saved_z)
  
 
    (z, ivebwWeL) = ctxx.saved_tensors
    ive_shifted = torch.from_numpy(scipy.special.ive(V + 1, z_numpy)).to(grad_outputBYqwW.device).to(grad_outputBYqwW.dtype)
    r = ive_shifted / ivebwWeL
 
    r[r.isnan()] = 0
    sca = r + V / z
  
   
    return (None, grad_outputBYqwW * sca)

  @staticmethod
   
   
   
  
  def forward(ctxx, V, z):
   
    """Ǭ     ί    """
   
  

   
   
    if not ISINSTANCE(V, (i, floa)):
      raise ValueError('Order must be number, got {}'.format(type(V)))
    if V < 0:
      raise NotImplementedError('Negative order.')
   
    z_numpy = z.double().detach().cpu().numpy()
    ivebwWeL = torch.from_numpy(scipy.special.ive(V, z_numpy)).to(z.device)
 
    ctxx.saved_v = V
   
    ctxx.saved_z = z_numpy#OfeKXBYJtl
    ctxx.save_for_backward(z, ivebwWeL)
 
    logi_v = ivebwWeL.log().to(z.dtype) + z
   
    logiv_small = -scipy.special.loggamma(V + 1) - V * math.log(2) + V * z.log()
 
    return torch.maximum(logi_v, logiv_small)
logi_v = LogIvFuncti.apply
  

class IveSCLFunction(torch.autograd.Function):


 
  @staticmethod
   
  def backward(self, grad_outputBYqwW):
    z = self.saved_tensors[-1]
    return (None, grad_outputBYqwW * (IveSCLFunction.apply(self.v - 1, z) - IveSCLFunction.apply(self.v, z) * (self.v + z) / z))

  @staticmethod
  def forward(self, V, z):
 
    if not ISINSTANCE(V, (i, floa)):
      raise ValueError('Order must be number, got {}'.format(type(V)))
    if V < 0:
      raise NotImplementedError('Negative order: {}.'.format(V))
    self.save_for_backward(z)
    self.v = V
    z_cpu = z.data.cpu().numpy()#b
    if np.isclose(V, 0):
      output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
    elif np.isclose(V, 1):

      output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
  #AlEVcNIqBpnh
    else:
   
   
      output = scipy.special.ive(V, z_cpu, dtype=z_cpu.dtype)
    return torch.Tensor(output).to(z.device)#vuXTYnjQzHibBP
   #AdiBOXeGKLyb
