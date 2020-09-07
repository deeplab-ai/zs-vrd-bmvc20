# -*- coding: utf-8 -*-
"""Model training/testing pipeline."""

import argparse
import json
import os

from common.models import load_model
from research.research_config import ResearchConfig
from research.src.models.sg_generator import (
    atr_net,
    language_net,
    lang_spat_net,
    reldn_net,
    spatial_net,
    visual_net,
    visual_spat_net,
    visual_spat_attention_net,
    uvtranse_net
)
from research.src.models.sg_projector import (
    language_projector,
    lang_spat_projector,
    spatial_projector,
    visual_projector,
    visual_spat_attention_projector,
    visual_spat_projector
)

MODELS = {
    'atr_net': atr_net,
    'language_net': language_net,
    'language_projector': language_projector,
    'lang_spat_net': lang_spat_net,
    'lang_spat_projector': lang_spat_projector,
    'reldn_net': reldn_net,
    'spatial_net': spatial_net,
    'spatial_projector': spatial_projector,
    'uvtranse_net': uvtranse_net,
    'visual_net': visual_net,
    'visual_projector': visual_projector,
    'visual_spat_attention_net': visual_spat_attention_net,
    'visual_spat_attention_projector': visual_spat_attention_projector,
    'visual_spat_net': visual_spat_net,
    'visual_spat_projector': visual_spat_projector
}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    # Model to train/test and peculiar parameters
    parser.add_argument(
        '--model', dest='model', help='Model to train (see main.py)',
        type=str, default='lang_spat_net'
    )
    parser.add_argument(
        '--model_params', dest='model_params',
        help='Dictionary of params peculiar to a model',
        type=str, default="{}"
    )
    parser.add_argument(
        '--object_classifier', dest='object_classifier',
        help='Name of classifier model to use if task is sgcls',
        type=str, default='object_classifier'
    )
    parser.add_argument(
        '--teacher', dest='teacher',
        help='Name of teacher model to use for distillation',
        type=str, default=None
    )
    parser.add_argument(
        '--teacher_name', dest='teacher_name',
        help='Name of teacher net (e.g. visual_net2_predcls_VRD)',
        type=str, default=None
    )
    # Dataset/task parameters
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset codename (e.g. VG200)',
        type=str, default='VRD'
    )
    parser.add_argument(
        '--task', dest='task',
        help='Task to solve, check config.py for supported tasks',
        type=str, default='preddet'
    )
    parser.add_argument(
        '--net_name', dest='net_name', help='Name of trained model',
        type=str, default=''
    )
    parser.add_argument(
        '--phrase_recall', dest='phrase_recall',
        help='Whether to evaluate phrase recall',
        action='store_true'
    )
    parser.add_argument(
        '--test_dataset', dest='test_dataset',
        help='Dataset to evaluate on, if different than train dataset',
        type=str, default=None
    )
    # Specific task parameters: data handling
    parser.add_argument(
        '--annotations_per_batch', dest='annotations_per_batch',
        help='Batch size in terms of annotations (e.g. relationships)',
        type=int, default=128
    )
    parser.add_argument(
        '--not_augment_annotations', dest='not_augment_annotations',
        help='Do not augment annotations with box/image distortion',
        action='store_true'
    )
    parser.add_argument(
        '--bg_perc', dest='bg_perc',
        help='Percentage of background annotations',
        type=float, default=None
    )
    parser.add_argument(
        '--filter_duplicate_rels', dest='filter_duplicate_rels',
        help='Whether to filter relations annotated more than once',
        action='store_true'
    )
    parser.add_argument(
        '--filter_multiple_preds', dest='filter_multiple_preds',
        help='Whether to sample a single predicate per object pair',
        action='store_true'
    )
    parser.add_argument(
        '--max_train_samples', dest='max_train_samples',
        help='Keep classes at most such many training samples',
        type=int, default=None
    )
    parser.add_argument(
        '--num_tail_classes', dest='num_tail_classes',
        help='Keep such many classes with the fewest training samples',
        type=int, default=None
    )
    # Evaluation parameters
    parser.add_argument(
        '--compute_accuracy', dest='compute_accuracy',
        help='For preddet only, measure accuracy instead of recall',
        action='store_true'
    )
    parser.add_argument(
        '--use_merged', dest='use_merged',
        help='Evaluate with merged predicate annotations',
        action='store_true'
    )
    # General model parameters
    parser.add_argument(
        '--is_not_context_projector', dest='is_not_context_projector',
        help='Do not treat this projector as a context projector',
        action='store_true'
    )
    parser.add_argument(
        '--is_cos_sim_projector', dest='is_cos_sim_projector',
        help='Maximize cos. similarity between features and learned weights',
        action='store_true'
    )
    # Specific task parameters: loss function
    parser.add_argument(
        '--not_use_multi_tasking', dest='not_use_multi_tasking',
        help='Do not use multi-tasking to detect "no interaction" cases',
        action='store_true'
    )
    parser.add_argument(
        '--use_weighted_ce', dest='use_weighted_ce',
        help='Use weighted cross-entropy',
        action='store_true'
    )
    # Training parameters
    parser.add_argument(
        '--batch_size', dest='batch_size',
        help='Batch size in terms of images',
        type=int, default=None
    )
    parser.add_argument(
        '--epochs', dest='epochs', help='Number of training epochs',
        type=int, default=None
    )
    parser.add_argument(
        '--learning_rate', dest='learning_rate',
        help='Learning rate of classification layers (not backbone)',
        type=float, default=0.002
    )
    parser.add_argument(
        '--weight_decay', dest='weight_decay',
        help='Weight decay of optimizer',
        type=float, default=None
    )
    # Learning rate policy
    parser.add_argument(
        '--apply_dynamic_lr', dest='apply_dynamic_lr',
        help='Adapt learning rate so that lr / batch size = const',
        action='store_true'
    )
    parser.add_argument(
        '--not_use_early_stopping', dest='not_use_early_stopping',
        help='Do not use early stopping learning rate policy',
        action='store_true'
    )
    parser.add_argument(
        '--not_restore_on_plateau', dest='not_restore_on_plateau',
        help='Do not restore best model on validation plateau',
        action='store_true'
    )
    parser.add_argument(
        '--patience', dest='patience',
        help='Number of epochs to consider a validation plateu',
        type=int, default=1
    )
    # Other data loader parameters
    parser.add_argument(
        '--commit', dest='commit',
        help='Commit name to tag model',
        type=str, default=''
    )
    parser.add_argument(
        '--num_workers', dest='num_workers',
        help='Number of workers employed by data loader',
        type=int, default=2
    )
    parser.add_argument(
        '--rel_batch_size', dest='rel_batch_size',
        help='Number of relations per sub-batch (memory issues)',
        type=int, default=128
    )
    return parser.parse_args()


def main():
    """Train and test a network pipeline."""
    args = parse_args()
    model = MODELS[args.model]
    _path = 'prerequisites/'
    cfg = ResearchConfig(
        dataset=args.dataset,
        task=args.task,
        net_name=args.net_name if args.net_name else args.model,
        phrase_recall=args.phrase_recall,
        test_dataset=args.test_dataset,
        annotations_per_batch=args.annotations_per_batch,
        augment_annotations=not args.not_augment_annotations,
        bg_perc=args.bg_perc,
        filter_duplicate_rels=args.filter_duplicate_rels,
        filter_multiple_preds=args.filter_multiple_preds,
        max_train_samples=args.max_train_samples,
        num_tail_classes=args.num_tail_classes,
        compute_accuracy=args.compute_accuracy,
        use_merged=args.use_merged,
        is_context_projector=not args.is_not_context_projector,
        is_cos_sim_projector=args.is_cos_sim_projector,
        use_multi_tasking=not args.not_use_multi_tasking,
        use_weighted_ce=args.use_weighted_ce,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        apply_dynamic_lr=args.apply_dynamic_lr,
        patience=args.patience,
        restore_on_plateau=not args.not_restore_on_plateau,
        use_early_stopping=not args.not_use_early_stopping,
        commit=args.commit,
        num_workers=args.num_workers,
        prerequisites_path=_path
    )
    model_params = json.loads('"' + args.model_params + '"')
    obj_classifier = None
    teacher = None
    model.train_test(cfg, obj_classifier, teacher, model_params)


if __name__ == "__main__":
    main()
