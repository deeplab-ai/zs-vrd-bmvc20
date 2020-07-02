# -*- coding: utf-8 -*-
"""A baseline using only linguistic features."""

import torch
from torch import nn

from .base_sg_generator import BaseSGGenerator


class LanguageNet(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config, features={}):
        """Initialize layers."""
        super().__init__(config, {})
        self.save_memory()
        self.fc_subject = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_object = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, self.num_rel_classes)
        )
        self.fc_classifier_bin = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
            self.get_obj_embeddings(objects['ids'][pairs[:, 0]]),
            self.get_obj_embeddings(objects['ids'][pairs[:, 1]])
        )

    def _forward(self, subj_embs, obj_embs):
        """Forward pass, returns output scores."""
        embeddings = torch.cat(
            (self.fc_subject(subj_embs), self.fc_object(obj_embs)),
            dim=1
        )
        scores = self.fc_classifier(embeddings)
        scores_bin = self.fc_classifier_bin(embeddings)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin
