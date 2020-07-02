# -*- coding: utf-8 -*-
"""A baseline using only visual and spatial features."""

import torch
from torch import nn

from .base_sg_generator import BaseSGGenerator


class VisualSpatNet(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config, features={}):
        """Initialize layers."""
        super().__init__(
            config, {'base_features', 'object_masks', 'pool_features'}
        )

        # Visual features
        self.fc_subject = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.fc_predicate = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.fc_object = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.pred_classifier = nn.Linear(512, self.num_rel_classes)
        self.os_classifier = nn.Linear(256, self.num_rel_classes)

        # Spatial features
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 8), nn.ReLU()
        )
        self.delta_net = nn.Sequential(
            nn.Linear(38, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.spatial_classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.num_rel_classes)
        )

        # Fusion
        self.fc_classifier = nn.Sequential(
            nn.Linear(1024 + 256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.num_rel_classes)
        )
        self.fc_classifier_bin = nn.Sequential(
            nn.Linear(1024 + 256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
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

    def _forward(self, subj_feats, pred_feats, obj_feats,
                 deltas, subj_masks, obj_masks):
        """Forward pass, returns output scores."""
        # Feature processing and deep scores
        subj_feats, pred_feats, obj_feats, pred_scores, os_scores = \
            self.visual_forward(subj_feats, pred_feats, obj_feats)
        spat_features, spat_scores = \
            self.spatial_forward(deltas, subj_masks, obj_masks)
        features = torch.cat(
            (subj_feats, pred_feats, obj_feats, spat_features), dim=1
        )

        # Classification
        scores = self.fc_classifier(features)
        scores_bin = self.fc_classifier_bin(features)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin, pred_scores, os_scores, spat_scores

    def spatial_forward(self, deltas, subj_masks, obj_masks):
        """Forward of spatial net."""
        masks = torch.cat((subj_masks, obj_masks), dim=1)
        features = torch.cat((
            self.mask_net(masks).view(masks.shape[0], -1),
            self.delta_net(deltas)
        ), dim=1)
        scores = self.spatial_classifier(features)
        return features, scores

    def visual_forward(self, subj_feats, pred_feats, obj_feats):
        """Forward of spatial net."""
        subj_feats = self.fc_subject(subj_feats)
        pred_feats = self.fc_predicate(pred_feats)
        obj_feats = self.fc_object(obj_feats)
        pred_scores = self.pred_classifier(pred_feats)
        os_scores = self.os_classifier(obj_feats - subj_feats)
        return subj_feats, pred_feats, obj_feats, pred_scores, os_scores
