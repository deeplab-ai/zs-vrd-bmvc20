# -*- coding: utf-8 -*-
"""A simple net using only spatial features."""

import torch
from torch import nn

from .base_sg_generator import BaseSGGenerator


class SpatialNet(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config, features={}):
        """Initialize layers."""
        super().__init__(config, {'object_masks'})
        self.save_memory()
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 8), nn.ReLU()
        )
        self.fc_delta = nn.Sequential(
            nn.Linear(38, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.fc_classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.num_rel_classes)
        )
        self.fc_classifier_bin = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
            self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]]
            ),
            objects['masks'][pairs[:, 0]],
            objects['masks'][pairs[:, 1]]
        )

    def _forward(self, deltas, subj_masks, obj_masks):
        """Forward pass."""
        masks = torch.cat((subj_masks, obj_masks), dim=1)
        features = torch.cat((
            self.mask_net(masks).view(masks.shape[0], -1),
            self.fc_delta(deltas)
        ), dim=1)
        scores = self.fc_classifier(features)
        scores_bin = self.fc_classifier_bin(features)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin
