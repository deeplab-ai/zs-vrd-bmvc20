# -*- coding: utf-8 -*-
"""A class to be inherited by other scene graph projectors."""

import torch
from torch import nn

from common.models.sg_generator import BaseSGGenerator


class BaseSGProjector(BaseSGGenerator):
    """Extends PyTorch nn.Module, base class for projectors."""

    def __init__(self, config, features):
        """Initialize layers."""
        super().__init__(config, features)
        self.class_projector = nn.Sequential(
            nn.Linear(300, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.weights = nn.Parameter(torch.rand(self.num_rel_classes, 128))
        # Contextualized predicate embeddings
        self.pre_rnn_proj_layer = nn.Sequential(
            nn.Linear(300, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.rnn_mixer = nn.GRU(128, 128, batch_first=True, bidirectional=True)

    def class_forward(self):
        """Forward of predicate class projector."""
        return self.class_projector(self.pred2vec)

    def context_forward(self, subj_embs, obj_embs):
        """Forward of local-context class projector."""
        subj_embs = self.pre_rnn_proj_layer(subj_embs)
        pred_embs = self.pre_rnn_proj_layer(self.pred2vec)
        obj_embs = self.pre_rnn_proj_layer(obj_embs)
        context_embs = []
        for subj_emb, obj_emb in zip(subj_embs, obj_embs):
            embs = torch.stack((
                subj_emb.unsqueeze(0).expand(len(pred_embs), -1),
                pred_embs,
                obj_emb.unsqueeze(0).expand(len(pred_embs), -1)
            ), dim=1)
            embs, _ = self.rnn_mixer(embs)
            context_embs.append(
                embs.view(len(embs), 3, 2, -1).sum(2).max(1)[0]
            )
        return torch.stack(context_embs)

    def get_classifiers(self, subj_embs, obj_embs):
        """Forward of local-context class projector."""
        if self.pred2vec is None:
            self._set_word2vec()
        if self.is_cos_sim_projector:
            return self.weights
        if self.is_context_projector:
            return self.context_forward(subj_embs, obj_embs)
        return self.class_forward()

    def reset_from_config(self, config):
        """Reset parameters from a config object."""
        super().reset_from_config(config)
        self.is_context_projector = config.is_context_projector
        self.is_cos_sim_projector = config.is_cos_sim_projector
