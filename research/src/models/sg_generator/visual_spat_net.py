# -*- coding: utf-8 -*-
"""A baseline using only visual and spatial features."""

from common.models.sg_generator import VisualSpatNet
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
        scores, _, pred_scores, os_scores, spat_scores = outputs
        targets = self.data_loader.get('predicate_ids', batch, step)

        # Losses
        losses = {
            'CE': self.criterion(scores, targets),
            'p-CE': self.criterion(pred_scores, targets),
            'os-CE': self.criterion(os_scores, targets),
            'spat-CE': self.criterion(spat_scores, targets)
        }
        loss = losses['CE']
        if self.training_mode:
            loss = (
                loss
                + 0.5 * losses['p-CE']
                + 0.5 * losses['os-CE']
                + 0.5 * losses['spat-CE']
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
    net = VisualSpatNet(config)
    train_tester = TrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()
