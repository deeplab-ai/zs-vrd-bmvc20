# -*- coding: utf-8 -*-
"""A baseline using only visual and spatial features."""

import torch
from torch import nn
from torch.nn import functional as F

from common.models.sg_generator import VisualSpatNet
from .base_sg_projector import BaseSGProjector


class VisualSpatProjector(BaseSGProjector, VisualSpatNet):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(
            config, {'base_features', 'object_masks', 'pool_features'}
        )
        self.fc_projector = nn.Sequential(
            nn.Linear(1024 + 256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
            self.get_obj_embeddings(objects['ids'][pairs[:, 0]]),
            self.get_obj_embeddings(objects['ids'][pairs[:, 1]]),
            objects['pool_features'][pairs[:, 0]],
            self.get_pred_pooled_features(
                base_features,
                objects['boxes'][pairs[:, 0]], objects['boxes'][pairs[:, 1]]
            ),
            objects['pool_features'][pairs[:, 1]],
            self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]]
            ),
            objects['masks'][pairs[:, 0]],
            objects['masks'][pairs[:, 1]]
        )

    def _forward(self, subj_embs, obj_embs, subj_feats, pred_feats, obj_feats,
                 deltas, subj_masks, obj_masks):
        """Forward pass, returns output scores."""
        # Feature processing and deep scores
        subj_feats, pred_feats, obj_feats, pred_scores, os_scores = \
            self.visual_forward(subj_feats, pred_feats, obj_feats)
        spat_features, spat_scores = \
            self.spatial_forward(deltas, subj_masks, obj_masks)
        fused_features = torch.cat(
            (subj_feats, pred_feats, obj_feats, spat_features), dim=1
        )

        # Classification
        features = self.fc_projector(fused_features)
        classifiers = self.get_classifiers(subj_embs, obj_embs)
        scores = F.cosine_similarity(classifiers, features.unsqueeze(1), dim=2)
        scores_bin = self.fc_classifier_bin(fused_features)
        if self.mode == 'test':
            scores = self.softmax(5 * scores)
            scores_bin = self.softmax(scores_bin)
        return (
            scores, scores_bin, classifiers,
            pred_scores, os_scores, spat_scores
        )
