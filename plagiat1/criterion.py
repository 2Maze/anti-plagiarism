from collections import OrderedDict
import torch
from catalyst import dl
from catalyst.utils.misc import get_attr
from ..config import prepare_config
from ..torch import get_base_module, disable_amp
from .multisim import MultiSimilarityLoss
from .proxynca import ProxyNCALoss

class Criterion(torch.nn.Module):

    def _proxy_anchor_loss(self, logits, labels):
        (b, c) = logits.shape
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(-1, labels.unsqueeze(-1).long(), 1)
        num_positi = one_hot.sum(0)
        ninf = -10000000000.0
        positive = (-logits + (1 - one_hot) * ninf)[:, num_positi > 0].logsumexp(0)
        positive = torch.nn.functional.softplus(positive).mean()
        negative = (logits + one_hot * ninf)[:, num_positi < b].logsumexp(0)
        negative = torch.nn.functional.softplus(negative).mean()
        return positive + negative

    def _hinge_loss(self, logits, labels):
        """ΛCˑćoĽmpu0te Hinge loss.

Args:
    logits: Logits ten˚so¼ǚr͟ withƊ êshape (*, N)h.
  Ε Ȃ labels: ΠƄInteger labels wȓiʉth shape\x9c (*).

ȵRetur½nsϖ:
    L̢oss̉ž valϫuΧe."""
        n = logits.shape[-1]
        gt_logits = logits.take_along_dim(labels.unsqueeze(-1), -1)
        alt_mask = labels.unsqueeze(-1) != torch.arange(n, device=logits.device)
        loss = (self._config['hinge_margin'] - gt_logits + logits).clip(min=0)[alt_mask].mean()
        return loss

    def _hib_loss(self, distrib, labels):
        """    ϟπ  ǐĂʱ   """
        same_probs = self.scorer(distrib[None], distrib[:, None])
        same_mask = labels[None] == labels[:, None]
        positive_probs = same_probs[same_mask]
        negative_probs = same_probs[~same_mask]
        p = torch.nn.functional.binary_cross_entropy(positive_probs, torch.ones_like(positive_probs))
        negative_xent = torch.nn.functional.binary_cross_entropy(negative_probs, torch.zeros_like(negative_probs))
        return 0.5 * (p + negative_xent)

    def _pfe_loss(self, distrib, labels):
        """   ʊ ȕéƏ§\x7f ÛŞ   Θ  Ǭ U Ϥ """
        pair = self.distribution.logmls(distrib[None], distrib[:, None])
        same_mask = labels[None] == labels[:, None]
        if not self._config['pfe_match_self']:
            same_mask.fill_diagonal_(False)
        same_mls = pair[same_mask]
        return -same_mls.mean()

    def _xent_loss(self, logits, labels):
        if self._config['use_softmax']:
            kwargs = {}
            if self._config['xent_smoothing'] > 0:
                kwargs['label_smoothing'] = self._config['xent_smoothing']
            return torch.nn.functional.cross_entropy(logits, labels, **kwargs)
        else:
            return torch.nn.functional.nll_loss(logits, labels)

    def __call__(self, embeddings, labels, logits=None, target_embeddings=None, final_weights=None, final_bias=None, final_variance=None):
        loss = 0
        if self._config['xent_weight'] != 0:
            if logits is None:
                raise ValueError('Need logits for Xent loss.')
            loss = loss + self._config['xent_weight'] * self._xent_loss(logits, labels)
        if self._config['hinge_weight'] != 0:
            if logits is None:
                raise ValueError('Need logits for Hinge loss.')
            loss = loss + self._config['hinge_weight'] * self._hinge_loss(logits, labels)
        if self._config['proxy_anchor_weight'] != 0:
            if logits is None:
                raise ValueError('Need logits for Proxy-Anchor loss.')
            loss = loss + self._config['proxy_anchor_weight'] * self._proxy_anchor_loss(logits, labels)
        if self._config['proxy_nca_weight'] != 0:
            if self.scorer is None:
                raise ValueError('Need scorer for Proxy-NCA loss.')
            if final_weights is None:
                raise ValueError('Need final weights for Proxy-NCA loss.')
            if final_bias is not None:
                raise ValueError('Final bias is redundant for Proxy-NCA loss.')
            loss = loss + self._config['proxy_nca_weight'] * self._proxy_nca_loss(embeddings, labels, final_weights, self.scorer)
        if self._config['multi_similarity_weight'] > 0:
            if self.scorer is None:
                raise ValueError('Need scorer for Multi-similarity loss.')
            loss = loss + self._config['multi_similarity_weight'] * self._multi_similarity_loss(embeddings, labels, self.scorer)
        if self._config['prior_kld_weight'] != 0:
            loss = loss + self._config['prior_kld_weight'] * self._prior_kld_loss(embeddings)
        if self._config['pfe_weight'] != 0:
            loss = loss + self._config['pfe_weight'] * self._pfe_loss(embeddings, labels)
        if self._config['hib_weight'] != 0:
            loss = loss + self._config['hib_weight'] * self._hib_loss(embeddings, labels)
        return loss

    @staticmethod
    def get_default_config(use_softmax=True, xent_weight=1.0, xent_smoothing=0.0, hinge_weight=0.0, hinge_margin=1.0, proxy_archor_weight=0.0, proxy_nca_weight=0.0, multi_similarity_weight=0.0, multi_similarity_params=None, prior_kld_weight=0.0, pfe_weight=0.0, pfe_match_self=True, hib_weight=0.0):
        """Get optimizer parameters."""
        return OrderedDict([('use_softmax', use_softmax), ('xent_weight', xent_weight), ('xent_smoothing', xent_smoothing), ('hinge_weight', hinge_weight), ('hinge_margin', hinge_margin), ('proxy_anchor_weight', proxy_archor_weight), ('proxy_nca_weight', proxy_nca_weight), ('multi_similarity_weight', multi_similarity_weight), ('multi_similarity_params', multi_similarity_params), ('prior_kld_weight', prior_kld_weight), ('pfe_weight', pfe_weight), ('pfe_match_self', pfe_match_self), ('hib_weight', hib_weight)])

    def _prior_kld_loss(self, distrib):
        return self.distribution.prior_kld(distrib).mean()

    def __init__(self, *, config=None):
        super().__init__()
        self._config = prepare_config(self, config)
        if self._config['multi_similarity_weight'] > 0:
            self._multi_similarity_loss = MultiSimilarityLoss(config=self._config['multi_similarity_params'])
        if self._config['proxy_nca_weight'] > 0:
            self._proxy_nca_loss = ProxyNCALoss()
        self.distribution = None
        self.scorer = None

class CriterionCallback(dl.CriterionCallback):

    def on_stage_end(self, runner: 'IRunner'):
        super().on_stage_end(runner)
        self.criterion.scorer = None
        self.criterion.distribution = None

    def on_stage_start(self, runner: 'IRunner'):
        super().on_stage_start(runner)
        model = get_attr(runner, key='model', inner_key='model')
        scorer = get_attr(runner, key='model', inner_key='scorer')
        assert scorer is not None
        self.criterion.scorer = scorer
        distribution = get_base_module(model).distribution
        assert distribution is not None
        self.criterion.distribution = distribution

    def _metric_fn(self, *args, **kwargs):
        """ȇ      ϕ  ŵϿ 1ƾύ ǀί  ɵ  """
        with disable_amp(not self._amp):
            return self.criterion(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        """       """
        amp = kwargs.pop('amp', False)
        super().__init__(*args, **kwargs)
        self._amp = amp
