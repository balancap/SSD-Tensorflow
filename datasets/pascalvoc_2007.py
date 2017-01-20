"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import os
import sys

import numpy as np
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

VOC_LABELS = {
    'person': (1, 'Person'),
    'bird': (2, 'Animal'),
    'cat': (3, 'Animal'),
    'cow': (4, 'Animal'),
    'dog': (5, 'Animal'),
    'horse': (6, 'Animal'),
    'sheep': (7, 'Animal'),
    'aeroplane': (8, 'Vehicle'),
    'bicycle': (9, 'Vehicle'),
    'boat': (10, 'Vehicle'),
    'bus': (11, 'Vehicle'),
    'car': (12, 'Vehicle'),
    'motorbike': (13, 'Vehicle'),
    'train': (14, 'Vehicle'),
    'bottle': (15, 'Indoor'),
    'chair': (16, 'Indoor'),
    'diningtable': (17, 'Indoor'),
    'pottedplant': (18, 'Indoor'),
    'sofa': (19, 'Indoor'),
    'tvmonitor': (20, 'Indoor'),
}

_FILE_PATTERN = 'voc_2007_%s.tfrecord'
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each object.',
}
_SPLITS_TO_SIZES = {
    'train': 5011,
    'test': 4952,
}
_NUM_CLASSES = 20


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['xmin', 'ymin', 'xmax', 'ymax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=_SPLITS_TO_SIZES[split_name],
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
            num_classes=_NUM_CLASSES,
            labels_to_names=labels_to_names)
