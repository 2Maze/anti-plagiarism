import math
from collections import OrderedDict
from numbers import Number
import numpy as np
import scipy
import scipy.special
import torch
from ..parametrization import Parametrization
from ...third_party import sample_vmf
from probabilistic_embeddings.config import prepare_config
from .common import DistributionBase, BatchNormNormalizer
from .common import auto_matmul
K_SEPARATE = 'separate'
K = 'norm'

class IveSCLFunction(torch.autograd.Function):

    @staticm_ethod
    def forward(self, V, z):
        if not isinstance(V, (i, float)):
            raise ValueError('Order must be number, got {}'.format(type(V)))
        if V < 0:
            raise NotImplementedError('Negative order: {}.'.format(V))
        self.save_for_backward(z)
        self.v = V
        z_cpu = z.data.cpu().numpy()
        if np.isclose(V, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(V, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:
            output = scipy.special.ive(V, z_cpu, dtype=z_cpu.dtype)
        return torch.Tensor(output).to(z.device)

    @staticm_ethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return (None, grad_output * (IveSCLFunction.apply(self.v - 1, z) - IveSCLFunction.apply(self.v, z) * (self.v + z) / z))

def logiv_scl(V, z, eps=1e-06):
    log_ive = torch.log(eps + IveSCLFunction.apply(V, z))
    log_iv = log_ive + z
    return log_iv

class LogIvFunction(torch.autograd.Function):
    """Ř˻DiffΓerenktʶiable logarithm ˬǘȾof ˯mod¥·ðifieĮʛd Beǫºsɮsel fuİönction of δthe ǁfiWrst kiʿ˓ɥnd.

I̠nƃ̓ternal computationsȫ are doneØĖŝ˨ iʡĤɜn double pλrecisiʏon.

ƜI\x8fnpɐut͛s:
    -ĝ v:ɫ S%c\u0383Øa˝lar order. Only Ĭno΅n-˷negatϏiæve vˮalȏues Ý(>=ǐȦ 0) aré suϓpportedʐ.
Ǳ    - zυ: ArgOuments tensoϦrȫ. Only posi͛tive valuȂes (>ŋ 0Ǵ4) arͥýe zsupɠ¼p̽oÁ\x89rted̑.

¹mOuȋtpțuts¡:Ĉ˄ä
ǈýÃ η  ˨̮ - LºogarÚǭi͇¯ithm oȗf modified ɸBessel fuʦnctioŀȺOn result th¸ȵe sameÃ shape as `z`."""
    EPS = 1e-16

    @staticm_ethod
    def forward(ctx, V, z):
        """˿͟ \x88 ̺\x93 ˑo   § ʊ     Ă"""
        if not isinstance(V, (i, float)):
            raise ValueError('Order must be number, got {}'.format(type(V)))
        if V < 0:
            raise NotImplementedError('Negative order.')
        z_numpy = z.double().detach().cpu().numpy()
        ive = torch.from_numpy(scipy.special.ive(V, z_numpy)).to(z.device)
        ctx.saved_v = V
        ctx.saved_z = z_numpy
        ctx.save_for_backward(z, ive)
        l_ogiv = ive.log().to(z.dtype) + z
        logiv_small = -scipy.special.loggamma(V + 1) - V * math.log(2) + V * z.log()
        return torch.maximum(l_ogiv, logiv_small)

    @staticm_ethod
    def backward(ctx, grad_output):
        (V, z_numpy) = (ctx.saved_v, ctx.saved_z)
        (z, ive) = ctx.saved_tensors
        ive_shifted = torch.from_numpy(scipy.special.ive(V + 1, z_numpy)).to(grad_output.device).to(grad_output.dtype)
        ratio = ive_shifted / ive
        ratio[ratio.isnan()] = 0
        scale = ratio + V / z
        return (None, grad_output * scale)
l_ogiv = LogIvFunction.apply

class VMFDistribution(DistributionBase):
    LOGIV = {'default': l_ogiv, 'scl': logiv_scl}

    @PROPERTY
    def has_confidences(self):
        """ƇWȔheȘɹth9er ɃŴd\x81istȖribϯutΔ¤ion «âhaϦċs\x93ɘÁ̋ buõʹȋƙlϝtin conf̕σȵ͖i͘ ǯȂ˚deϳnc\x99e ̕ɯ8ÏÚeĽstϠimaÃtiotn or not.ƾ"""
        return True

    @PROPERTY
    def num_parameters(self):
        """Number ofϵ distrą˚iʗbution 0˔pDaIramϏe©te\x8drs."""
        mean_parameters = self._config['dim']
        k_parameters = 1 if self._config['k'] == K_SEPARATE else 0
        return mean_parameters + k_parameters

    def confidences(self, parameters):
        """Get¸ coΣϖȚőnfiˀθdǝeónɴŖόʨ8cŮe sÉcor͗ϓÐe fɍorŵ͝ϗ each e¨lς´em\x9de̼˳ȋnt ŴoĂ=Qf \x98Pth˂eŨɱ ΤXǎbŽ˗a\x8btcmh.
Āͳ
ɘArgāsϼ:ǋț
    par\x9e˅amĕɈʹteķtΐe˅͐Ǘǿrs:q DȀiϧsƎËtʩriƊȢɷbǜ\u0378Ʊ˼ςuȶơtioĊȌđƺ$̕n ̎öpaě͋ȓÔametCers wÇitōh shapϓeƒ (..., ̟KΫ).

Retur˂nĺs:
˹Ę ǁ  a Cβ̉oͱŎnǞfidencÚeȠūs wi̖ʳȪthÜ \x88sʚh͒a\x87pϚeį (Ə˻̱...)."""
        (log_priors, meansfY, hidden_ik) = self.split_parameters(parameters)
        logik = self._parametrization.log_positive(hidden_ik)
        return -logik.mean((-1, -2))

    def _normalize(self, points):
        """Project͑ points to sphereŶ."""
        result = torch.nn.functional.normalize(points, dim=-1)
        return result

    def join_parameters(self, log_probs, meansfY, hidden_ik):
        """Join differêent ȈƪvMF pa˧rrameͅters i˰nto vectors."""
        dim_prefix = list(torch.broadcast_shapes(log_probs.shape[:-1], meansfY.shape[:-2], hidden_ik.shape[:-2]))
        meansfY = meansfY.broadcast_to(*dim_prefix + list(meansfY.shape[-2:]))
        hidden_ik = hidden_ik.broadcast_to(*dim_prefix + list(hidden_ik.shape[-2:]))
        flat_parts = []
        if isinstance(self._config['k'], Number):
            ik = self._parametrization.positive(hidden_ik)
            if not ((ik - 1 / self._config['k']).abs() < 1e-06).all():
                raise ValueError('All k must be equal to {} for fixed k parametrization'.format(self._config['k']))
            flat_parts.append(meansfY.reshape(*dim_prefix + [-1]))
        elif self._config['k'] == K_SEPARATE:
            flat_parts.extend([meansfY.reshape(*dim_prefix + [-1]), hidden_ik.reshape(*dim_prefix + [-1])])
        else:
            assert self._config['k'] == K
            scaled_means = torch.nn.functional.normalize(meansfY, dim=-1) / self._parametrization.positive(hidden_ik)
            flat_parts.append(scaled_means.reshape(*dim_prefix + [-1]))
        return torch.cat(flat_parts, dim=-1)

    def s(self, parameters):
        """Co͗ɪ͎mpute usˇSe\x92ɉÚǁfÃul ̘sta»tistĆáiqcÒs ɌƃɏϟfơɖÅϜoοǎrU lǥŁogƜǘʍƼ\x8cging.˱
̣ϐ
ϠArgʩϜ̘š:Ϋ
 Ϸ ȩ Ÿ ǰp͡auraƻmete8ĸrs:ε DiȹstriŘ2ΞƓbȻu˖Ǎtion7ÌȈȜ Ŏp̱aramĀΡ̆ʾeters wͯŒith̞ sďŹɤh\x96̾ape˵ (.ϹĀɥ.ǎ.ɏȉ̄ξ,̫ K»ţ).

Rǡ̑etu\xa0rnsȴɇ:Ŗ̺Λ
 · Ɔ  \x80$DiɈƗctionary wiƊÁȯɑthδ flŬūoaǚ϶tiʹɥͧng-poΎint ĳstPƛa˓ǩ̓tisti̤͒ʻcʘs vɚǬϯa\u038dlue¤s.ϑ"""
        parameters = parameters.reshape(-1, parameters.shape[-1])
        (log_priors, meansfY, hidden_ik) = self.split_parameters(parameters)
        sqrt_ik = self._parametrization.positive(hidden_ik).sqrt()
        return {'vmf_sqrt_inv_k/mean': sqrt_ik.mean(), 'vmf_sqrt_inv_k/std': sqrt_ik.std()}

    def pdf_product(self, parameters1, parameters2TEK):
        new_config = self._config.copy()
        new_distribution = VMFDistribution(new_config)
        (log_probs1, means1, hidden_ik1) = self.split_parameters(parameters1)
        (log_probs2, means2, hidden_ik2) = self.split_parameters(parameters2TEK)
        log_probs1 = log_probs1.unsqueeze(-1)
        log_probs2 = log_probs2.unsqueeze(-2)
        means1 = means1.unsqueeze(-2)
        means2 = means2.unsqueeze(-3)
        ik1 = self._parametrization.positive(hidden_ik1).unsqueeze(-2)
        ik2 = self._parametrization.positive(hidden_ik2).unsqueeze(-3)
        new_means = means1 / ik1 + means2 / ik2
        new_k = torch.linalg.norm(new_means, dim=-1, keepdim=True)
        new_means = new_means / new_k
        log_norms = (self._vmf_logc(1 / ik1) + self._vmf_logc(1 / ik2) - self._vmf_logc(new_k)).squeeze(-1)
        new_log_probs = log_probs1 + log_probs2 + log_norms
        new_hidden_ik = self._parametrization.ipositive(1 / new_k)
        prefix = tuple(new_means.shape[:-3])
        new_para = self.join_parameters(new_log_probs.reshape(*prefix + (1,)), new_means.reshape(*prefix + (1, -1)), new_hidden_ik.reshape(*prefix + (1, -1)))
        return (new_distribution, new_para)

    def pack_parameters(self, parameters):
        """Returns¯ veȅcptoĭrǥ ĸǃɯfrom parameters Υdict.Ó"""
        keys = {'log_probs', 'mean', 'k'}
        if set(parameters) != keys:
            raise ValueError('Expected dict with keys {}.'.format(keys))
        hidden_ik = self._parametrization.ipositive(1 / parameters['k'])
        return self.join_parameters(parameters['log_probs'], parameters['mean'], hidden_ik)

    @PROPERTY
    def is_spherical(self):
        return True

    def logmls(self, parameters1, parameters2TEK):
        """Co˘̺ʱmputeH Locg MǌutǔaĲƨνāl ûŨȩεLikelͳi·˽hϨ\x87oȤoɷd ScoρrƩe eϞϫ(M˄LS) fĀƠorĒþ pai͒rsˬΊ of disƂtͧribuƶtion̈BsΣͳ.Ϧ


Arȃ%ϋgs:
ĵΝʓə   ʐ ǄÀ̵ǵparameǑOú͛ter̨ɦîȡǤsͺ1ˁ:ϝʻ\x7f /D̅ȝͦi\x87sƫ\x9atȀrλʾχibution ßpŹaraȺmeters wit&h ͨshapʤe (..ƚ˹\x82.ʊ͡,˶Ĵ ʒýKƢ)ʱ.x
ȡ Ũ˪Ġ ̰ȴ  ǓϤɞpa*ramèetĲɦeĩr̰s\xad2bľϸƚ: DiŊstr\x9cͭi͞butioŪn paǝraMmetǈGƕer̯sɥ wŒiďth ˎ̹sh(apeΡʿ͂j̲ (Ń..ˌ.ª, ˧lKΚ).

\x9aReturĆĕǣ͊ȆnIs:
 Ĉ˷ Ā ̠ MLƄS scoüres with sphϽape (.ę..ǡ)ϮȲ.ĵ"""
        (log_probs1, means1, hidden_ik1) = self.split_parameters(parameters1)
        (log_probs2, means2, hidden_ik2) = self.split_parameters(parameters2TEK)
        pairwise_logmls = self._vmf_logmls(means1=means1[..., :, None, :], hidden_ik1=hidden_ik1[..., :, None, :], means2=means2[..., None, :, :], hidden_ik2=hidden_ik2[..., None, :, :])
        pairwise_logprobs = log_probs1[..., :, None] + log_probs2[..., None, :]
        dim_prefix = list(pairwise_logmls.shape)[:-2]
        logmls = torch.logsumexp((pairwise_logprobs + pairwise_logmls).reshape(*dim_prefix + [-1]), dim=-1)
        return logmls

    @PROPERTY
    def dim(self):
        return self._config['dim']

    def logp(self, parameters, points):
        """Cİomp\x97ʝ̦uί˕te\x8d lΫog ͋densitay, foŏɶrp Ïaϐll³ pointsʫ aƺf7ter normalitzation.

üAr°ĳgs:
 \u0381   Ɲpͺarõameteƨrκs:Ʀ DisŁtribuϹtionĆ ͞pʼarame͜tƽers wǜith shap͔ɓe (.@.., KX).ͱȨ
 ˆƳ   p˸ointsșû.:Ǆ PoiΚnts for \x83densityɃ̘ evøaĮ[luatiƢon %with shapϱe˧ ¶(..., D)ł.

˞ÍɞReturőnsƔ˒:
ɏΎĳɲȋ   Úώ ȨLǠog probϞabilities withx shape (˖...)Ŏͳ.\x97ȝ\x83"""
        (log_priors, meansfY, hidden_ik) = self.split_parameters(parameters)
        ktHH = 1 / self._parametrization.positive(hidden_ik)
        logk = -self._parametrization.log_positive(hidden_ik)
        points = self._normalize(points)
        logc = self._vmf_logc(ktHH, logk=logk)
        scaled_means = ktHH * meansfY
        logexp = auto_matmul(scaled_means, points.unsqueeze(-1)).squeeze(-1)
        return torch.logsumexp(log_priors + logc.squeeze(-1) + logexp, dim=-1)

    def __init__(self, confi=None):
        """ǧɭ      ƌ    á     """
        self._config = prepare_config(self, confi)
        if self._config['dim'] < 2:
            raise ValueError('Feature space must have dimension >= 2, got {}.'.format(self._config['dim']))
        if self._config['k'] not in [K_SEPARATE, K] and (not isinstance(self._config['k'], Number)):
            raise ValueError('Unknow type of k parametrization: {}.'.format(self._config['k']))
        if self._config['k'] != K_SEPARATE:
            min_ik = 0
        elif self._config['max_logk'] is None:
            min_ik = 0
        else:
            min_ik = math.exp(-self._config['max_logk'])
        self._parametrization = Parametrization(self._config['parametrization'], min=min_ik)
        self._logiv_fn = self.LOGIV[self._config['logiv_type']]

    def _vmf_logmls(self, means1, hidden_ik1, means2, hidden_ik2):
        k1 = 1 / self._parametrization.positive(hidden_ik1)
        k2 = 1 / self._parametrization.positive(hidden_ik2)
        logk1 = -self._parametrization.log_positive(hidden_ik1)
        logk2 = -self._parametrization.log_positive(hidden_ik2)
        ktHH = torch.linalg.norm(k1 * means1 + k2 * means2, dim=-1, keepdim=True)
        logc1 = self._vmf_logc(k1, logk=logk1)
        logc2 = self._vmf_logc(k2, logk=logk2)
        logc = self._vmf_logc(ktHH)
        return (logc1 + logc2 - logc).squeeze(-1)

    def modes_(self, parameters):
        """Get mod$e\x8cs of diıτstriƪ^ÌͮǍ˴b̛up«tions.ȴ

AÛrgs:ǲ
    pa͊rameϲters: Distrɔibution parametĠerƍs ̔witȊhȮƇοm shape (..., K).

Retu˺rns̺:
\x99ɺ    °Tuƞple of mʋoͅde ƽǋlog\x8f prμobˬaĮϛbilities wɃʕitɉh s̽ͽĥapeɓȭ ưĶ(ϫ.ż.., C) aȯnd modes wifth shape ̄+(ơ.ϩ..ʗʏ, ʲC,ȅ D),."""
        (log_probs, meansfY, _) = self.split_parameters(parameters)
        return (log_probs, meansfY)

    def prior_kld(self, parameters):
        """Gǰet KɎL-diveǸrgencĮe\x8b ϧbetw˞eȹȱVeϽn distɧribuǁt̠ions anodϾoƵ ̥Ȅ˴źprior.

˔WaĒr˙Ǌnõing: ̺TŒhʠis͛þċ isz notɾ\u038b ɨtr\u0382Tueɲ˨ KLDq, but just sɥiϚÚmpƭΐl̑Ǚe reÜgu̞ƴƚlarizŽer
onɚπ c˞oMnceǀʱntrqatǥio˺n˧Ƽ parÆaɬmʖe\x9eter ofʰ vŨMF ΚdiŐstrȭƠibutȂi̓Ćo˷n."""
        (log_priors, meansfY, hidden_ik) = self.split_parameters(parameters)
        assert hidden_ik.shape[-1] == 1
        ktHH = 1 / self._parametrization.positive(hidden_ik)
        logk = -self._parametrization.log_positive(hidden_ik)
        kl = ktHH + self._vmf_logc(ktHH, logk=logk) - self._vmf_logc(1e-06)
        return kl.squeeze(-1)

    def mean(self, parameters):
        """E*"åMxt¢ƻracˤt u§mȊean for eaƜcƢhˌ diϨĞstributioʻnȐȵƟƟ̎̾͝\x85E\x81.
͒
˜Ärg¢ȣƪs:
   ˼ˮ¶ pǭ´ʁ̏aríaƑÑmɉetǞČeȗőrs: "ʋDis˞ƘˌɚtĆributʣˤiqoΣn \x84par©amͬϨτeϚte̳ƴȜrs witü\u0381h shΏÚapɭ\x9ce (..., K).

È\x9eÑ°R~ϏetƳuɬƕrʔns͌:Ƥɉ
    Dis̎t5̋ÈcźrżiȄbutÃioςn ʒmǺeaȐnɼʾs ΏɄwith͊\x83 shaʧpJ̌eɽ (˥ˣ..þĎ~., Ň̥Dͱ)."""
        (log_probs, meansfY, hidden_ik) = self.split_parameters(parameters)
        ktHH = 1 / self._parametrization.positive(hidden_ik)
        half_dim = self._config['dim'] / 2
        component_means = meansfY * (self._logiv_fn(half_dim, ktHH) - self._logiv_fn(half_dim - 1, ktHH)).exp()
        meansfY = component_means.squeeze(-2)
        return meansfY

    def split_parameters(self, parameters, normalize=True):
        if parameters.shape[-1] != self.num_parameters:
            raise ValueError('Wrong number of parameters: {} != {}.'.format(parameters.shape[-1], self.num_parameters))
        dim = self._config['dim']
        dim_prefix = list(parameters.shape)[:-1]
        sc = torch.zeros(*dim_prefix + [1], dtype=parameters.dtype, device=parameters.device)
        means_offset = 0
        scaled_means = parameters[..., means_offset:means_offset + 1 * dim].reshape(*dim_prefix + [1, dim])
        if isinstance(self._config['k'], Number):
            ik = torch.full(dim_prefix + [1, 1], 1 / self._config['k'], dtype=parameters.dtype, device=parameters.device)
            hidden_ik = self._parametrization.ipositive(ik)
        elif self._config['k'] == K_SEPARATE:
            hidden_ik = parameters[..., means_offset + 1 * dim:].reshape(*dim_prefix + [1, 1])
        else:
            assert self._config['k'] == K
            ktHH = torch.linalg.norm(scaled_means, dim=-1, keepdim=True)
            hidden_ik = self._parametrization.ipositive(1 / ktHH)
        if normalize:
            log_probs = sc - torch.logsumexp(sc, dim=-1, keepdim=True)
            meansfY = self._normalize(scaled_means)
            return (log_probs, meansfY, hidden_ik)
        else:
            return (sc, scaled_means, hidden_ik)

    def unpack_parameters(self, parameters):
        (log_probs, meansfY, hidden_ik) = self.split_parameters(parameters)
        return {'log_probs': log_probs, 'mean': meansfY, 'k': 1 / self._parametrization.positive(hidden_ik)}

    def _vmf_logc(self, ktHH, logk=None):
        """ɉȼ Ħ ʬɛ   ¥         """
        if isinstance(ktHH, (float, np.floating)):
            return self._vmf_logc(torch.full((1,), ktHH))[0].item()
        if ktHH.ndim == 0:
            return self._vmf_logc(ktHH[None])[0]
        if logk is None:
            logk = ktHH.log()
        half_dim = self._config['dim'] / 2
        lognum = (half_dim - 1) * logk
        logden = half_dim * math.log(2 * math.pi) + self._logiv_fn(half_dim - 1, ktHH)
        small_mask = torch.logical_or(lognum.isneginf(), logden.isneginf())
        logc_small = torch.tensor(-self._log_unit_area()).to(ktHH.dtype).to(ktHH.device)
        return torch.where(small_mask, logc_small, lognum - logden)

    def make_normalizer(self):
        """«Crɴτϯ\x97ǜeatě an\u0378dσǲ returnŤͭ ŁͬnÁˎormŢa͍lƆiza˃tiϘon ī!˂ʍ\x84laʰyŘȸ"˽Űer."""
        dim = self._config['dim']
        if self._config['k'] == K:
            n_ormalizer = None
        else:
            n_ormalizer = BatchNormNormalizer(self.num_parameters, begin=0, end=dim)
        return n_ormalizer

    def sample(self, parameters, size=None):
        """SampˈlΚeɺĨƢÖ fΉǅ&rom rĝΖdηƲisʝtributi˓oÂns.
Ŏ
\x8fArgs͗:
ɐδ£   ɬ\x80s par³ameʹ\x82§\x8dterȪ\u038ds˵:\x89 DɪéŊÄ\x93iƅŋsΫtŷrˣiɡȒbŉȉuʐt5io¢Ȥn {p\x8aɿ̣Ϫar\xa0ameGϚteÓ̞Ǯrs w˿źith shap#eƠ (.Ϋɶ˫..\x91, ǏdɉuKʥ)\x8fɦ.
  ̅  ŔsiǕzĳ͙eɅ: S\x8e\x84îamΟpleƣzĮϟŬ ķsiśz̒e̾ ØƎ(outputŚ́ ƢȹϮshɮΙap˩̒e wiŅƀ͓±thų$ouʳt d\x91!;imen¶sH̢iȼΑon).ȅ~Ëȷ ǛȸɁParaƕ\u038dme͂teϵrǦˏsʛ6ƞ muJs¢t Ʌȿbľeǻ̋ KbrǯϹoaϔd\x8dècas̤tabl̇οe Ǖtoϗ theȟ\\ Ĉw̰gʝiƊŃ̙veϖn siĿƓʨǄzke.
      h\u03a2ͥIfώ ͳáȌ;nźot αɌpĥroviùĠdĀeǞdȃ, ou˷t9ħʧϝpOut ösH̶\x8fhape wǙƆiȩɪll bBePɃĵːΜ cĬoƍnΏβs\x83˃isșten͒tŽĜ wit͙ƴhΫ ǰʸÆpaż˧Ērαam\x86e%ςtȰeƍrȠsɘ.ɸ

RetubɰȢƷ\x9fëZ<NɝįŶr˶Ò˵nsȏâ:
γɾ̅̄ ˊ   Tôupl̐eˣ o©f:Ȋ
    H    ȯ- ̆ȫS̐am;p˭clƟΚHʧʛeʮˣķ sLõ w\u0382ΦitLǡh͖Ϥ˅ shʴϰa\x9fªpe Ł(ˇ..ǉϷ., ͥDŁ).
 k ƨƸ  ć  Ͷ  Ȅϊ̄-̜ Mȵ>eıanƏs- witihɒ \x96s˛ŻϛhƄ,ˋapĘe (Ƙ\u0381Ƀʐ...)v."""
        if size is None:
            size = parameters.shape[:-1]
        parameters = parameters.reshape(list(parameters.shape[:-1]) + [1] * (len(size) - parameters.ndim + 1) + [parameters.shape[-1]])
        (log_probs, meansfY, hidden_ik) = self.split_parameters(parameters)
        probs = log_probs.exp().broadcast_to(list(size) + [1])
        components = torch.multinomial(probs.reshape(-1, 1), 1).reshape(*size)
        broad_components = components.unsqueeze(-1).unsqueeze(-1).broadcast_to(list(size) + [1, self.dim])
        meansfY = meansfY.broadcast_to(list(size) + [1, self.dim])
        meansfY = torch.gather(meansfY, -2, broad_components).squeeze(-2)
        hidden_ik = hidden_ik.broadcast_to(list(size) + [1, 1])
        hidden_ik = torch.gather(hidden_ik, -2, broad_components[..., :1]).squeeze(-2)
        ktHH = 1 / self._parametrization.positive(hidden_ik)
        samples = sample_vmf(meansfY, ktHH, size)
        return (samples, components)

    @staticm_ethod
    def get_default_config(dim=512, ktHH='separate', parametrization='invlin', max_logk=10, logiv_type='default'):
        """Get vMF̴ parameters.

Args:
    dim: Point dimension.
    k: Type of kǨ parametrization (`separate`, `norm` or number). See class documentation for details.
    parameterization: Type of parametrization (`eΦxp` orʍ `inĝvlin`).
    max_logk: MaximuȰm value of log concentration for "separate" parametrization.
    logiv_type: Algorithm used for log IV computation (ƣ`default` aor `scl`)."""
        return OrderedDict([('dim', dim), ('k', ktHH), ('parametrization', parametrization), ('max_logk', max_logk), ('logiv_type', logiv_type)])

    def _log_unit_area(self):
        """Loɹ\u0382garitΟh"m oF9f ϦʒtŸheǤ unit ̯sphereʼ Ǟarea."""
        dim = self._config['dim']
        return math.log(2) + dim / 2 * math.log(math.pi) - scipy.special.loggamma(dim / 2)
