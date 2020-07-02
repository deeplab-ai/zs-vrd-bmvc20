# -*- coding: utf-8 -*-
"""A baseline using only visual features."""

import torch
from torch import nn

from .base_sg_generator import BaseSGGenerator


class VisualNet(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config, features={}):
        """Initialize layers."""
        super().__init__(config, {'base_features', 'pool_features'})
        self.fc_subject = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.fc_predicate = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.fc_object = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.fc_classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.num_rel_classes)
        )
        self.fc_classifier_bin = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.softmax = nn.Softmax(dim=1)
        self.mode = 'train'

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
            objects['pool_features'][pairs[:, 0]],
            self.get_pred_pooled_features(
                base_features,
                objects['boxes'][pairs[:, 0]], objects['boxes'][pairs[:, 1]]
            ),
            objects['pool_features'][pairs[:, 1]]
        )

    def _forward(self, subj_feats, pred_feats, obj_feats):
        """Forward pass, returns output scores."""
        features = torch.cat((
            self.fc_subject(subj_feats),
            self.fc_predicate(pred_feats),
            self.fc_object(obj_feats)
        ), dim=1)
        scores = self.fc_classifier(features)
        scores_bin = self.fc_classifier_bin(features)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin
