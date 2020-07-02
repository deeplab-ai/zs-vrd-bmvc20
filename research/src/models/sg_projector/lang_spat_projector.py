# -*- coding: utf-8 -*-
"""A baseline using only linguistic and spatial features."""

from common.models.sg_projector import LangSpatProjector
from research.src.train_testers import SGProjTrainTester


def train_test(config, obj_classifier=None, teacher=None, model_params={}):
    """Train and test a net."""
    net = LangSpatProjector(config)
    train_tester = SGProjTrainTester(
        net, config, set(), obj_classifier, teacher
    )
    train_tester.train_test()
