# -*- coding: utf-8 -*-
"""
A class for training/testing a network on Scene Graph Generation.

Methods _compute_loss and _net_outputs assume that _net_forward
returns (pred_scores, rank_scores, ...).
They should be re-implemented if that's not the case.
"""

import json
import random

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from research.src.evaluators import (
    RankingClsEvaluator, RelationshipClsEvaluator, RelationshipEvaluator
)
from research.src.data_loaders import SGGDataset, SGGDataLoader
from common.tools import AnnotationLoader

from .base_train_tester_class import BaseTrainTester


class SGGTrainTester(BaseTrainTester):
    """
    Train and test utilities for Scene Graph Generation.

    Inputs upon initialization:
        - net: PyTorch nn.Module, the network to train/test
        - config: class Config, see config.py
        - features: set of str, features to load for net
        - obj_classifier: ObjectClassifier object (see corr. API)
        - teacher: a loaded SGG model
    """

    def __init__(self, net, config, features, obj_classifier=None,
                 teacher=None):
        """Initiliaze train/test instance."""
        super().__init__(net, config, features)
        self.obj_classifier = obj_classifier
        self.teacher = teacher

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        # Settings and loading
        self.logger.info(
            "Test %s on %s on %s" % (self._net_name, self._task, self._dataset)
        )
        self.training_mode = False
        self.net.eval()
        self.net.to(self._device)
        self._set_data_loaders(mode_ids={'test': 2})
        self.data_loader = self._data_loaders['test']
        if self._task == 'predcls' and self._use_multi_tasking:
            rank_eval = RankingClsEvaluator(self.annotation_loader)
        elif self._compute_accuracy or self._dataset in {'VG80K', 'UnRel'}:
            rel_eval = RelationshipClsEvaluator(self.annotation_loader,
                                                self._use_merged)
        else:
            rel_eval = RelationshipEvaluator(self.annotation_loader,
                                             self._use_merged)

        # Forward pass on test set
        results = {}
        for batch in tqdm(self.data_loader):
            for step in range(len(batch['filenames'])):
                boxes = batch['boxes'][step]  # detected boxes
                labels = batch['labels'][step]  # store (s,p,o) labels here
                pred_scores, rank_scores, subj_scores, obj_scores = \
                    self._net_outputs(batch, step)
                scores = pred_scores.cpu().numpy()
                if self._task not in {'preddet', 'predcls'}:
                    subj_scores = subj_scores.cpu().numpy()
                    obj_scores = obj_scores.cpu().numpy()
                    scores = (
                        np.max(subj_scores, axis=1)[:, None]
                        * scores
                        * np.max(obj_scores, axis=1)[:, None])
                    labels[:, 0] = np.argmax(subj_scores, axis=1)
                    labels[:, 2] = np.argmax(obj_scores, axis=1)
                filename = batch['filenames'][step]
                if self._classes_to_keep is not None:
                    scores[:, self._classes_to_keep] += 10
                rel_eval.step(filename, scores, labels, boxes,
                              self._phrase_recall)
                if rank_scores is not None and self._task == 'predcls':
                    rank_eval.step(filename, rank_scores.cpu().numpy())
                results.update({  # top-5 classification results
                    filename: {
                        'scores': np.sort(scores)[:, ::-1][:, :5].tolist(),
                        'classes': scores.argsort()[:, ::-1][:, :5].tolist()
                    }})
        # Print metrics and save results
        rel_eval.print_stats(self._task)
        if rank_scores is not None and self._task == 'predcls':
            rank_eval.print_stats()
        end = '_tail.json' if self._classes_to_keep is not None else '.json'
        if self._dataset not in self._net_name:
            end = end.replace('.', '_%s.' % self._dataset)
        with open(self._results_path + 'results' + end, 'w') as fid:
            json.dump(results, fid)

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self._net_forward(batch, step)
        scores = outputs[0]
        targets = self.data_loader.get('predicate_ids', batch, step)

        # Losses
        losses = {'CE': self.criterion(scores, targets)}
        loss = losses['CE']

        # Knowledge Distillation
        if self._use_multi_tasking and self._task != 'preddet':
            loss += self._multitask_loss(outputs[1], batch, step)
        if self.teacher is not None:
            losses['KD'] = self._kd_loss(scores, outputs[1], batch, step)
            if self.training_mode:
                loss += losses['KD']
        return loss, losses

    @torch.no_grad()
    def _kd_loss(self, scores, bg_scores, batch, step):
        """Compute knowledge distillation loss."""
        self.teacher.eval()
        t_outputs = self.teacher(
            self.data_loader.get('images', batch, step),
            self.data_loader.get('object_rois', batch, step),
            self.data_loader.get('object_ids', batch, step),
            self.data_loader.get('pairs', batch, step),
            self.data_loader.get('image_info', batch, step)
        )
        kd_loss = 25 * F.kl_div(
            F.log_softmax(scores / 5, 1), F.softmax(t_outputs[0] / 5, 1),
            reduction='none'
        ).mean(1)
        if self._use_multi_tasking and self._task != 'preddet':
            kd_loss += 5 * F.kl_div(
                F.log_softmax(bg_scores, 1), F.softmax(t_outputs[1], 1),
                reduction='none'
            ).mean(1)
        return kd_loss

    def _multitask_loss(self, bg_scores, batch, step):
        """Reformulate loss to involve bg/fg multi-tasking."""
        bg_targets = self.data_loader.get('bg_targets', batch, step)
        return F.cross_entropy(bg_scores, bg_targets, reduction='none')

    def _net_forward(self, batch, step):
        """Return a tuple of scene graph generator's outputs."""
        return self.net(
            self.data_loader.get('images', batch, step),
            self.data_loader.get('object_rois', batch, step),
            self.data_loader.get('object_ids', batch, step),
            self.data_loader.get('pairs', batch, step),
            self.data_loader.get('image_info', batch, step)
        )

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch."""
        obj_vecs = self._object_forward(batch, step)
        pairs = self.data_loader.get('pairs', batch, step)
        s_scores = obj_vecs[pairs[:, 0]]
        o_scores = obj_vecs[pairs[:, 1]]
        rest_outputs = self._net_forward(batch, step)
        p_scores = rest_outputs[0]
        if self._task == 'preddet' or not self._use_multi_tasking:
            return p_scores, None, s_scores, o_scores
        bg_scores = rest_outputs[1]
        return (
            p_scores * bg_scores[:, 1].unsqueeze(-1),
            bg_scores, s_scores, o_scores
        )

    def _object_forward(self, batch, step, base_feats=None):
        """Return object vectors for different tasks."""
        if self.net.mode == 'train' or self._task != 'sgcls':
            obj_ids = self.data_loader.get('object_ids', batch, step)
            obj_vecs = torch.zeros(len(obj_ids), self._num_obj_classes)
            obj_vecs = obj_vecs.to(self._device)
            obj_vecs[torch.arange(len(obj_ids)), obj_ids] = 1.0
        elif self._task == 'sgcls':
            obj_vecs = self.obj_classifier(
                self.data_loader.get('images', batch, step),
                self.data_loader.get('object_rois', batch, step),
                self.data_loader.get('object_ids', batch, step),
                self.data_loader.get('image_info', batch, step)
            )
        return obj_vecs

    def _set_data_loaders(self, mode_ids={'train': 0, 'val': 1, 'test': 2}):
        annotations = np.array(self.annotation_loader.get_annos())
        split_ids = np.array([anno['split_id'] for anno in annotations])
        datasets = {
            split: SGGDataset(
                annotations[split_ids == split_id].tolist(),
                self.config, self.features)
            for split, split_id in mode_ids.items()
        }
        self._data_loaders = {
            split: SGGDataLoader(
                datasets[split], batch_size=self._batch_size,
                shuffle=split == 'train', num_workers=self._num_workers,
                drop_last=split != 'test', device=self._device)
            for split in mode_ids
        }
