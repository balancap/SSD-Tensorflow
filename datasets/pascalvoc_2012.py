# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import tensorflow as tf
from datasets import pascalvoc_common

slim = tf.contrib.slim

FILE_PATTERN = 'voc_2012_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (670, 865),
    'bicycle': (552, 711),
    'bird': (765, 1119),
    'boat': (508, 850),
    'bottle': (706, 1259),
    'bus': (421, 593),
    'car': (1161, 2017),
    'cat': (1080, 1217),
    'chair': (1119, 2354),
    'cow': (303, 588),
    'diningtable': (538, 609),
    'dog': (1286, 1515),
    'horse': (482, 710),
    'motorbike': (526, 713),
    'person': (4087, 8566),
    'pottedplant': (527, 973),
    'sheep': (325, 813),
    'sofa': (507, 566),
    'train': (544, 628),
    'tvmonitor': (575, 784),
    'total': (11540, 27450),
}
SPLITS_TO_SIZES = {
    'train': 17125,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
}
NUM_CLASSES = 20


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
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return pascalvoc_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)

