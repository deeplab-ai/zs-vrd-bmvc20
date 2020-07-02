# -*- coding: utf-8 -*-
"""A baseline using only linguistic features."""

import torch
from torch import nn
from torch.nn import functional as F

from common.models.sg_generator import LanguageNet
from .base_sg_projector import BaseSGProjector


class LanguageProjector(BaseSGProjector, LanguageNet):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(config, {})
        self.save_memory()
        self.fc_projector = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )

    def _forward(self, subj_embs, obj_embs):
        """Forward pass, returns output scores."""
        embeddings = torch.cat(
            (self.fc_subject(subj_embs), self.fc_object(obj_embs)),
            dim=1
        )
        features = self.fc_projector(embeddings)
        classifiers = self.get_classifiers(subj_embs, obj_embs)
        scores = F.cosine_similarity(classifiers, features.unsqueeze(1), dim=2)
        scores_bin = self.fc_classifier_bin(embeddings)
        if self.mode == 'test':
            scores = self.softmax(5 * scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin, classifiers
