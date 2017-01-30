"""Definition of VGG-based SSD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
  with slim.arg_scope(ssd_vgg.ssd_vgg()):
    outputs, end_points = ssd_vgg.ssd_vgg(inputs)
@@ssd_vgg
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

from nets import custom_layers

slim = tf.contrib.slim


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(21, 45),
                      (45, 99),
                      (99, 153),
                      (153, 207),
                      (207, 261),
                      (261, 315)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
        """Network definition.
        """
        return ssd_net(inputs,
                       num_classes=self.params.num_classes,
                       feat_layers=self.params.feat_layers,
                       anchor_sizes=self.params.anchor_sizes,
                       anchor_ratios=self.params.anchor_ratios,
                       normalizations=self.params.normalizations,
                       is_training=is_training,
                       dropout_keep_prob=dropout_keep_prob,
                       prediction_fn=prediction_fn,
                       reuse=reuse,
                       scope=scope)

    def arg_scope(self, weight_decay=0.0005):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (len(n_feat_layers)-2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * 0.07, img_size * 0.15]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]

    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]
    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], scope='conv_loc')
    loc_pred = tf.reshape(loc_pred, tf.concat(0, [loc_pred.get_shape()[:-1],
                                                  [num_anchors],
                                                  [4]]))
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], scope='conv_cls')
    cls_pred = tf.reshape(cls_pred, tf.concat(0, [cls_pred.get_shape()[:-1],
                                                  [num_anchors],
                                                  [num_classes]]))
    return cls_pred, loc_pred


def ssd_net(inputs,
            num_classes=21,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
    """SSD net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs]):
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3')
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3')
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions, localisations, logits, end_points
ssd_net.default_image_size = 300


def ssd_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME') as sc:
            return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc
