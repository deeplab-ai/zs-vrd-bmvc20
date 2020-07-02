# -*- coding: utf-8 -*-
"""A simple net using only spatial features."""

from common.models.sg_generator import SpatialNet
from research.src.train_testers import SGGTrainTester


def train_test(config, obj_classifier=None, teacher=None, model_params={}):
    """Train and test a net."""
    net = SpatialNet(config)
    train_tester = SGGTrainTester(net, config, set(), obj_classifier, teacher)
    train_tester.train_test()
