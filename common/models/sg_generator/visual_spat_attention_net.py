# -*- coding: utf-8 -*-
"""A net using visual/spatial features and attention."""

import torch
from torch import nn

from .base_sg_generator import BaseSGGenerator


class VisualSpatAttentionNet(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config, features={}):
        """Initialize layers."""
        super().__init__(
            config,
            {'base_features', 'object_masks', 'pool_features', 'roi_features'}
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

        # Attentive visual features
        self.conv_subject = nn.Sequential(nn.Conv2d(256, 256, 1), nn.ReLU())
        self.conv_predicate = nn.Sequential(nn.Conv2d(256, 512, 1), nn.ReLU())
        self.conv_object = nn.Sequential(nn.Conv2d(256, 256, 1), nn.ReLU())
        self.vis_pooling_sobj = VisAttentionalPooling(256)
        self.vis_pooling_pred = VisAttentionalPooling(512)
        self.att_sobj_classifier = nn.Linear(256, self.num_obj_classes)
        self.att_pred_classifier = nn.Linear(512, self.num_rel_classes)
        self.fc_lang_projector = nn.Sequential(nn.Linear(600, 300), nn.Tanh())

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
            objects['roi_features'][pairs[:, 0]],
            self.get_roi_features(
                base_features,
                objects['boxes'][pairs[:, 0]], objects['boxes'][pairs[:, 1]]
            ),
            objects['roi_features'][pairs[:, 1]],
            objects['pool_features'][pairs[:, 0]],
            self.get_pred_pooled_features(
                base_features,
                objects['boxes'][pairs[:, 0]], objects['boxes'][pairs[:, 1]]
            ),
            objects['pool_features'][pairs[:, 1]],
            self.get_obj_embeddings(objects['ids'][pairs[:, 0]]),
            self.get_obj_embeddings(objects['ids'][pairs[:, 1]]),
            self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]]
            ),
            objects['masks'][pairs[:, 0]],
            objects['masks'][pairs[:, 1]]
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
        features = torch.cat(
            (subj_feats, pred_feats, obj_feats, spat_features), dim=1
        )

        # Classification
        scores = self.fc_classifier(features)
        scores_bin = self.fc_classifier_bin(features)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return (
            scores, scores_bin,
            pred_scores, os_scores, spat_scores,
            att_subj_scores, att_pred_scores, att_obj_scores
        )

    def attention_forward(self, subj_feats, pred_feats, obj_feats,
                          subj_embs, obj_embs):
        """Forward of attentive visual net."""
        subj_feats, _ = self.vis_pooling_sobj(
            self.conv_subject(subj_feats), subj_embs)
        pred_feats, _ = self.vis_pooling_pred(
            self.conv_predicate(pred_feats),
            self.fc_lang_projector(torch.cat((subj_embs, obj_embs), dim=1)))
        obj_feats, _ = self.vis_pooling_sobj(
            self.conv_object(obj_feats), obj_embs)
        subj_scores = self.att_sobj_classifier(subj_feats)
        pred_scores = self.att_pred_classifier(pred_feats)
        obj_scores = self.att_sobj_classifier(obj_feats)
        return (
            subj_feats, pred_feats, obj_feats,
            subj_scores, pred_scores, obj_scores
        )

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
        """Forward of visual net."""
        subj_feats = self.fc_subject(subj_feats)
        pred_feats = self.fc_predicate(pred_feats)
        obj_feats = self.fc_object(obj_feats)
        pred_scores = self.pred_classifier(pred_feats)
        os_scores = self.os_classifier(obj_feats - subj_feats)
        return subj_feats, pred_feats, obj_feats, pred_scores, os_scores


class VisAttentionalPooling(nn.Module):
    """Visual attentional pooling implementation."""

    def __init__(self, input_dim):
        """Initialize layers."""
        super().__init__()
        self.vis_projector = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh()
        )
        self.lang_projector = nn.Sequential(
            nn.Linear(300, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh()
        )
        self.cos_similarity = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_map, embeddings):
        """Forward pass."""
        feature_map = feature_map.permute(0, 2, 3, 1)
        feature_map = feature_map.view(len(feature_map), 196, -1)
        features = self.vis_projector(feature_map)
        attention = self.lang_projector(embeddings)
        attention_map = self.cos_similarity(features, attention.unsqueeze(1))
        soft_attention_map = self.softmax(attention_map).unsqueeze(2)
        return (feature_map * soft_attention_map).sum(dim=1), attention_map
