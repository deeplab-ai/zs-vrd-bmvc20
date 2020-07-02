# -*- coding: utf-8 -*-
"""Relationship Detection Network by Zhang et al., 2019."""

from common.models.sg_generator import RelDN
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
        scores = outputs[0]
        vis_scores, spat_scores = outputs[2:4]
        targets = self.data_loader.get('predicate_ids', batch, step)

        # Losses
        losses = {
            'CE': self.criterion(scores, targets),
            'vis-CE': self.criterion(vis_scores, targets),
            'spat-CE': self.criterion(spat_scores, targets)
        }
        loss = losses['CE'] + losses['vis-CE'] + losses['spat-CE']
        if self._use_multi_tasking and self._task != 'preddet':
            loss += self._multitask_loss(outputs[1], batch, step)
        if self.teacher is not None:
            losses['KD'] = self._kd_loss(scores, outputs[1], batch, step)
            if self.training_mode:
                loss += losses['KD']
        return loss, losses


def train_test(config, obj_classifier=None, teacher=None, model_params={}):
    """Train and test a net."""
    net = RelDN(config)
    train_tester = TrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()
