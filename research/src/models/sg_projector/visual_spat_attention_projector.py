# -*- coding: utf-8 -*-
"""A net using visual/spatial features and attention."""

import torch
from torch.nn import functional as F

from common.models.sg_projector import VisualSpatAttentionProjector
from research.src.train_testers import SGProjTrainTester


class TrainTester(SGProjTrainTester):
    """Extends SGProjTrainTester."""

    def __init__(self, net, config, features, obj_classifier, teacher):
        """Initialize instance."""
        super().__init__(net, config, features, obj_classifier, teacher)

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self._net_forward(batch, step)
        (
            scores, _, classifiers, pred_scores, os_scores, spat_scores,
            att_subj_scores, att_pred_scores, att_obj_scores
        ) = outputs
        targets = self.data_loader.get('predicate_ids', batch, step)
        object_ids = self.data_loader.get('object_ids', batch, step)
        pairs = self.data_loader.get('pairs', batch, step)
        subj_targets = object_ids[pairs[:, 0]]
        obj_targets = object_ids[pairs[:, 1]]
        gt_sims = self.data_loader.get('predicate_similarities', batch, step)

        # Loss: CE
        losses = {
            'CE': self.criterion(scores, targets),
            'p-CE': self.criterion(pred_scores, targets),
            'os-CE': self.criterion(os_scores, targets),
            'spat-CE': self.criterion(spat_scores, targets),
            'att_subj': F.cross_entropy(
                att_subj_scores, subj_targets, reduction='none'),
            'att_pred': self.criterion(att_pred_scores, targets),
            'att_obj': F.cross_entropy(
                att_obj_scores, obj_targets, reduction='none')
        }

        # Loss: language synonymy MSE
        sims = F.cosine_similarity(  # predicate similarities given context!
            classifiers,
            classifiers[torch.arange(len(targets)), targets, :].unsqueeze(1),
            dim=2
        )
        to_consider = (gt_sims > -1).float()
        losses['sims'] = 50 * F.mse_loss(
            sims * to_consider,
            gt_sims * to_consider
        )

        # Loss: visual synonymy kl-divergence
        losses['kl'] = 1000 * F.kl_div(
            F.log_softmax(scores, 1),
            F.softmax(sims, 1),
            reduction='none'
        ).mean(1)

        # Total loss
        loss = (
            losses['CE'] * (0.5 if self._epoch > 3 else 1)
            + losses['kl'] * (1 if self._epoch > 3 else 0.1)
            + losses['sims']
        )
        if self.training_mode:
            loss = (
                loss
                + 0.5 * losses['p-CE']
                + 0.5 * losses['os-CE']
                + 0.5 * losses['spat-CE']
                + 0.5 * losses['att_subj']
                + 0.5 * losses['att_pred']
                + 0.5 * losses['att_obj']
            )
        if self._use_multi_tasking and self._task != 'preddet':
            loss += self._multitask_loss(outputs[1], batch, step)
        if self.teacher is not None:
            losses['KD'] = self._kd_loss(scores, outputs[1], batch, step)
            if self.training_mode:
                loss += losses['KD']
        return loss, losses


def train_test(config, obj_classifier=None, teacher=None, model_params={}):
    """Train and test a net."""
    net = VisualSpatAttentionProjector(config)
    train_tester = TrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()
