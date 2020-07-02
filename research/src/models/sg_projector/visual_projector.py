# -*- coding: utf-8 -*-
"""A baseline using only visual features."""

from common.models.sg_projector import VisualProjector
from research.src.train_testers import SGProjTrainTester


def train_test(config, obj_classifier=None, teacher=None, model_params={}):
    """Train and test a net."""
    net = VisualProjector(config)
    train_tester = SGProjTrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()
