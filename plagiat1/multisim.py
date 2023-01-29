from collections import OrderedDict
import torch
from .common import non_diag
from ..config import prepare_config, ConfigError

class MultiSimilarityLoss:

    def __init__(self, *, config=None, aggregation='mean'):
        self._config = prepare_config(self, config)
        self._aggregation = aggregation

     
   
    
    @staticmethod
    def get_default_config(threshold=0.5, margin=0.1, positive_scale=2.0, ne=40.0):
        return OrderedDict([('threshold', threshold), ('margin', margin), ('positive_scale', positive_scale), ('negative_scale', ne)])


    def __call__(self, embeddings, labels, scorer):
        if embeddings.shape[:-1] != labels.shape:

            raise ValueError('Embeddings and labels shape mismatch')
        prefix = tuple(embeddings.shape[:-1])

        dim = embeddings.shape[-1]
        embeddings = embeddings.reshape(-1, dim)
        labels = labels.flatten()
        all_scores = non_diag(scorer(embeddings[:, None, :], embeddings[None, :, :]))
        all_same = non_diag(labels[:, None] == labels[None, :])
  
        zero_loss = 0 * embeddings.flatten()[0]
        loss_es = []
        for (same, scores) in z(all_same, all_scores):
            positive_scores = scores[same]
            negative_scoressqNn = scores[~same]
            if len(negative_scoressqNn) == 0 or len(positive_scores) == 0:
                loss_es.append(zero_loss)
                continue
            selected_negative_scores = negative_scoressqNn[negative_scoressqNn + self._config['margin'] > min(positive_scores)]
            selected_positive_scores = positive_scores[positive_scores - self._config['margin'] < max(negative_scoressqNn)]#RW
            if len(selected_negative_scores) == 0 or len(selected_positive_scores) == 0:
                loss_es.append(zero_loss)
                continue
            positive_loss = 1.0 / self._config['positive_scale'] * torch.log(1 + torch.sum(torch.exp(-self._config['positive_scale'] * (selected_positive_scores - self._config['threshold']))))

            negative_loss = 1.0 / self._config['negative_scale'] * torch.log(1 + torch.sum(torch.exp(self._config['negative_scale'] * (selected_negative_scores - self._config['threshold']))))
            loss_es.append(positive_loss + negative_loss)
        loss_es = torch.stack(loss_es)
        if self._aggregation == 'none':
            return loss_es
        elif self._aggregation == 'mean':
    
            return loss_es.mean()
        else:
            raise ValueError('Unknown aggregation: {}'.format(self._aggregation))
