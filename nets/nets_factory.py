# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""


import functools
import tensorflow as tf

from nets import inception
# from nets import overfeat
# from nets import resnet_v1
# from nets import resnet_v2
from nets import vgg
from nets import xception
from nets import ssd_vgg

slim = tf.contrib.slim

networks_map = {'vgg_a': vgg.vgg_a,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19,
                'inception_v3': inception.inception_v3,
                'inception_resnet_v2': inception.inception_resnet_v2,
                'xception': xception.xception,
                'ssd_300_vgg': ssd_vgg.ssd_300_vgg,
                'ssd_300_vgg_caffe': ssd_vgg.ssd_300_vgg,
                }

arg_scopes_map = {'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                  'inception_resnet_v2': inception.inception_resnet_v2_arg_scope,
                  'xception': xception.xception_arg_scope,
                  'ssd_300_vgg': ssd_vgg.ssd_300_vgg_arg_scope,
                  'ssd_300_vgg_caffe': ssd_vgg.ssd_300_vgg_caffe_scope,
                  }


def get_network_fn(name, num_classes, is_training=False, **kwargs):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      is_training: `True` if the model is being used for training and `False`
        otherwise.
      weight_decay: The l2 coefficient for the model weights.
    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature: logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    arg_scope = arg_scopes_map[name](**kwargs)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
