# -*- coding: utf-8 -*-
"""Custom dataset for Scene Graph Generation."""

from .base_data_loader import BaseDataset


class SGGDataset(BaseDataset):
    """Dataset utilities for Scene Graph Generation."""

    def __init__(self, annotations, config, features={}):
        """Initialize dataset."""
        super().__init__(annotations, config, features)
        self._features = features.union({
            'bg_targets', 'boxes', 'image_info', 'labels',
            'object_ids', 'object_rois', 'object_rois_norm',
            'pairs', 'predicate_ids'
        })
        if config.task not in {'preddet', 'predcls'}:
            self._features.add('images')
