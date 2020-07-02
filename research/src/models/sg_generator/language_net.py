# -*- coding: utf-8 -*-
"""A baseline using only linguistic features."""

from common.models.sg_generator import LanguageNet
from research.src.train_testers import SGGTrainTester


def train_test(config, obj_classifier=None, teacher=None, model_params={}):
    """Train and test a net."""
    net = LanguageNet(config)
    train_tester = SGGTrainTester(net, config, set(), obj_classifier, teacher)
    train_tester.train_test()
