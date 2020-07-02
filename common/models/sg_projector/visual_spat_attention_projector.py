# -*- coding: utf-8 -*-
"""A net using visual/spatial features and attention."""

import torch
from torch import nn
from torch.nn import functional as F

from common.models.sg_generator import VisualSpatAttentionNet
from .base_sg_projector import BaseSGProjector


class VisualSpatAttentionProjector(BaseSGProjector, VisualSpatAttentionNet):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(
            config,
            {'base_features', 'object_masks', 'pool_features', 'roi_features'}
        )
        self.fc_projector = nn.Sequential(
            nn.Linear(1024 + 256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )

    def _forward(self, subj_feats, pred_feats, obj_feats,  # pre-pooled
                 pld_subj_feats, pld_pred_feats, pld_obj_feats,  # pooled
                 subj_embs, obj_embs, deltas, subj_masks, obj_masks):
        """Forward pass, returns output scores."""
        # Feature processing and deep scores
        (
            att_subj_feats, att_pred_feats, att_obj_feats,
            att_subj_scores, att_pred_scores, att_obj_scores
        ) = self.attention_forward(
            subj_feats, pred_feats, obj_feats, subj_embs, obj_embs
        )
        subj_feats, pred_feats, obj_feats, pred_scores, os_scores = \
            self.visual_forward(pld_subj_feats, pld_pred_feats, pld_obj_feats)
        subj_feats = subj_feats + att_subj_feats
        pred_feats = pred_feats + att_pred_feats
        obj_feats = obj_feats + att_obj_feats
        spat_features, spat_scores = \
            self.spatial_forward(deltas, subj_masks, obj_masks)
        cat_features = torch.cat(
            (subj_feats, pred_feats, obj_feats, spat_features), dim=1
        )

        # Classification
        features = self.fc_projector(cat_features)
        classifiers = self.get_classifiers(subj_embs, obj_embs)
        scores = F.cosine_similarity(classifiers, features.unsqueeze(1), dim=2)
        scores_bin = self.fc_classifier_bin(cat_features)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return (
            scores, scores_bin, classifiers,
            pred_scores, os_scores, spat_scores,
            att_subj_scores, att_pred_scores, att_obj_scores
        )
