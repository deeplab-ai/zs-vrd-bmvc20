# -*- coding: utf-8 -*-
"""A baseline using only linguistic and spatial features."""

import torch
from torch import nn

from .base_sg_generator import BaseSGGenerator


class LangSpatNet(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config, features={}, **kwargs):
        """Initialize layers."""
        super().__init__(config, {'object_masks'})
        self.save_memory()

        # Language layers
        self.fc_subject = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_object = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_classifier_lang = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, self.num_rel_classes)
        )
        self.fc_classifier_lang_bin = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

        # Spatial layers
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
        self.fc_classifier_spatial = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.num_rel_classes)
        )
        self.fc_classifier_spatial_bin = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
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
        # Language classification
        embeddings = torch.cat(
            (self.fc_subject(subj_embs), self.fc_object(obj_embs)),
            dim=1)
        lang_scores = self.fc_classifier_lang(embeddings)
        lang_scores_bin = self.fc_classifier_lang_bin(embeddings)

        # Spatial classification
        masks = torch.cat((subj_masks, obj_masks), dim=1)
        spatial_features = torch.cat((
            self.mask_net(masks).view(masks.shape[0], -1),
            self.fc_delta(deltas)
        ), dim=1)
        spatial_scores = self.fc_classifier_spatial(spatial_features)
        spatial_scores_bin = self.fc_classifier_spatial_bin(spatial_features)

        # Fusion
        scores = lang_scores + spatial_scores
        scores_bin = lang_scores_bin + spatial_scores_bin
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin
