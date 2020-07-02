# -*- coding: utf-8 -*-
"""A simple net using only spatial features."""

import torch
from torch import nn
from torch.nn import functional as F

from common.models.sg_generator import SpatialNet
from .base_sg_projector import BaseSGProjector


class SpatialProjector(BaseSGProjector, SpatialNet):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(config, {'object_masks'})
        self.save_memory()
        self.fc_projector = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
            self.get_obj_embeddings(objects['ids'][pairs[:, 0]]),
            self.get_obj_embeddings(objects['ids'][pairs[:, 1]]),
            self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]]
            ),
            objects['masks'][pairs[:, 0]],
            objects['masks'][pairs[:, 1]]
        )

    def _forward(self, subj_embs, obj_embs, deltas, subj_masks, obj_masks):
        """Forward pass, returns output scores."""
        masks = torch.cat((subj_masks, obj_masks), dim=1)
        spatial = torch.cat((
            self.mask_net(masks).view(masks.shape[0], -1),
            self.fc_delta(deltas)
        ), dim=1)
        features = self.fc_projector(spatial)
        classifiers = self.get_classifiers(subj_embs, obj_embs)
        scores = F.cosine_similarity(classifiers, features.unsqueeze(1), dim=2)
        scores_bin = self.fc_classifier_bin(spatial)
        if self.mode == 'test':
            scores = self.softmax(5 * scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin, classifiers
