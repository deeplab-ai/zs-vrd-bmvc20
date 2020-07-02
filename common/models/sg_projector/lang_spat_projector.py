# -*- coding: utf-8 -*-
"""A baseline using only linguistic and spatial features."""

import torch
from torch import nn
from torch.nn import functional as F

from common.models.sg_generator import LangSpatNet
from .base_sg_projector import BaseSGProjector


class LangSpatProjector(BaseSGProjector, LangSpatNet):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(config, {'object_masks'})
        self.save_memory()
        self.fc_projector_lang = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.fc_projector_spatial = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )

    def _forward(self, subj_embs, obj_embs, deltas, subj_masks, obj_masks):
        """Forward pass, returns output scores."""
        # Language classification
        embeddings = torch.cat(
            (self.fc_subject(subj_embs), self.fc_object(obj_embs)),
            dim=1)
        lang_features = self.fc_projector_lang(embeddings)
        lang_scores_bin = self.fc_classifier_lang_bin(embeddings)

        # Spatial classification
        masks = torch.cat((subj_masks, obj_masks), dim=1)
        spatial = torch.cat((
            self.mask_net(masks).view(masks.shape[0], -1),
            self.fc_delta(deltas)
        ), dim=1)
        spatial_features = self.fc_projector_spatial(spatial)
        spatial_scores_bin = self.fc_classifier_spatial_bin(spatial)

        # Fusion
        features = lang_features + spatial_features
        classifiers = self.context_forward(subj_embs, obj_embs)
        scores = F.cosine_similarity(classifiers, features.unsqueeze(1), dim=2)
        scores_bin = lang_scores_bin + spatial_scores_bin
        if self.mode == 'test':
            scores = self.softmax(5 * scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin, classifiers
