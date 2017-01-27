"""Definition of VGG-based SSD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
  with slim.arg_scope(ssd_vgg.ssd_vgg()):
    outputs, end_points = ssd_vgg.ssd_vgg(inputs)
@@ssd_vgg
"""
import math
import numpy as np
import tensorflow as tf

from nets import custom_layers

slim = tf.contrib.slim


# =========================================================================== #
# VGG based SSD300 parameters.
# =========================================================================== #
ssd_300_features = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
ssd_300_features_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
# ssd_300_sizes = [[.1, 0.15], [.2, .276], [.38, .461], [.56, .644], [.74, .825], [.92, 1.01]]
ssd_300_sizes_limits = [0.15, 0.90]
# SSD ratios: note, we omit ratio which is always added by default.
ssd_300_ratios = [[2, .5],
                  [2, .5, 3, 1./3],
                  [2, .5, 3, 1./3],
                  [2, .5, 3, 1./3],
                  [2, .5],
                  [2, .5]]
ssd_300_normalizations = [20, -1, -1, -1, -1, -1]


def ssd_reference_sizes():
    """Compute the reference sizes of the anchor boxes.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    default_dim = ssd_300_vgg.default_image_size
    min_ratio = int(ssd_300_sizes_limits[0] * 100)
    max_ratio = int(ssd_300_sizes_limits[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (len(ssd_300_features)-2)))
    # Start with the following smallest sizes.
    sizes = [[default_dim * 0.07, default_dim * 0.15]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append([default_dim * ratio / 100.,
                      default_dim * (ratio + step) / 100.])
    return sizes


def ssd_anchor_boxes(img_shape, feat_shape,
                     sizes, ratios,
                     offset=0.5, dtype=np.float32):
    """Computer ssd default boxes.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      img_shape: Image shape, used for computing height width relatively to the
        former;
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      offset: Offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid.
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) / feat_shape[0]
    x = (x.astype(dtype) + offset) / feat_shape[1]

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


def ssd_anchors_from_layers(img_shape, layers_shape,
                            layers_sizes, layers_ratios,
                            offset=0.5, dtype=np.float32):
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_boxes(img_shape, s,
                                         layers_sizes[i], layers_ratios[i],
                                         offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# VGG based SSD300 implementation.
# =========================================================================== #
def ssd_multibox_layer(inputs, num_classes, sizes, ratios=[1],
                       normalization=-1, bn_normalization=False,
                       clip=True, interm_layer=0):
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
    loc_pred = tf.reshape(loc_pred, tf.concat(0, [tf.shape(loc_pred)[:-1],
                                                  [num_anchors],
                                                  [4]]))
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], scope='conv_cls')
    cls_pred = tf.reshape(cls_pred, tf.concat(0, [tf.shape(cls_pred)[:-1],
                                                  [num_anchors],
                                                  [num_classes]]))
    return cls_pred, loc_pred


def ssd_300_vgg(inputs,
                num_classes=21,
                is_training=True,
                dropout_keep_prob=0.5,
                prediction_fn=slim.softmax,
                reuse=None,
                scope='ssd_300_vgg'):
    """SSD model from https://arxiv.org/abs/1512.02325

    Features layers with 300x300 input:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1

    The default image size used to train this network is 300x300.
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
        ssd_300_sizes = ssd_reference_sizes()
        print(ssd_300_sizes)

        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(ssd_300_features):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          ssd_300_sizes[i],
                                          ssd_300_ratios[i],
                                          ssd_300_normalizations[i],
                                          clip=True, interm_layer=0)
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions, localisations, logits, end_points
ssd_300_vgg.default_image_size = 300


def ssd_300_vgg_arg_scope(weight_decay=0.0005):
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
def ssd_300_vgg_caffe_scope(caffe_scope):
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
