# -*- coding: utf-8 -*-
"""A net using visual/spatial features and attention."""

from torch.nn import functional as F

from common.models.sg_generator import VisualSpatAttentionNet
from research.src.train_testers import SGGTrainTester


class TrainTester(SGGTrainTester):
    """Extends SGGTrainTester."""

    def __init__(self, net, config, features, obj_classifier, teacher):
        """Initialize instance."""
        super().__init__(net, config, features, obj_classifier, teacher)

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self._net_forward(batch, step)
        (
            scores, _, pred_scores, os_scores, spat_scores,
            att_subj_scores, att_pred_scores, att_obj_scores
        ) = outputs
        targets = self.data_loader.get('predicate_ids', batch, step)
        object_ids = self.data_loader.get('object_ids', batch, step)
        pairs = self.data_loader.get('pairs', batch, step)
        subj_targets = object_ids[pairs[:, 0]]
        obj_targets = object_ids[pairs[:, 1]]

        # Losses
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
        loss = losses['CE']
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
    # config.reset()
    net = VisualSpatAttentionNet(config)
    train_tester = TrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()
