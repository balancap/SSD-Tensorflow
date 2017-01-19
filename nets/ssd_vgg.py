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

slim = tf.contrib.slim


def ssd_multibox_layer(inputs, num_classes, size, ratio=[1],
                       normalization=-1, bn_normalization=False,
                       clip=True, interm_layer=0):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = tf.div(net, normalization)
    # Number of anchors.
    num_anchors = len(size) + len(ratio) - 1

    # Class and location predictions.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], scope='conv_cls')
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], scope='conv_loc')
    return cls_pred, loc_pred


def ssd_default_boxes(feat_shape, size, ratio, dtype=np.float32):
    """Computer ssd default boxes. Center, width and height.
    """
    # Similarly to SSD paper: ratio only applies to first size parameter.
    num_anchors = len(size) + len(ratio) - 1
    boxes = np.array((*feat_shape, num_anchors, 4))
    for i in range(feat_shape[0]):
        for j in range(feat_shape[1]):
            for k, r in enumerate(ratio):
                s = size[0]
                boxes[i, j, k, :] = [(i+0.5) / feat_shape[0],
                                     (j+0.5) / feat_shape[1],
                                     s * math.sqrt(r),
                                     s / math.sqrt(r)]
            # Single box with different size.
            s = size[1]
            boxes[i, j, -1, :] = [(i+0.5) / feat_shape[0],
                                  (j+0.5) / feat_shape[1],
                                  s, s]
    return boxes

# =========================================================================== #
# VGG based SSD300 implementation.
# =========================================================================== #
ssd_300_features = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
ssd_300_features_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
ssd_300_sizes = [[.1], [.2, .276], [.38, .461], [.56, .644], [.74, .825], [.92, 1.01]]
ssd_300_ratios = [[1, 2, .5],
                  [1, 2, .5, 3, 1./3],
                  [1, 2, .5, 3, 1./3],
                  [1, 2, .5, 3, 1./3],
                  [1, 2, .5, 3, 1./3],
                  [1, 2, .5, 3, 1./3]]
ssd_300_normalizations = [20, -1, -1, -1, -1, -1]


def ssd_300_vgg(inputs,
                num_classes=1000,
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
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        end_points['block1'] = net
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        end_points['block2'] = net
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        end_points['block3'] = net
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        end_points['block4'] = net
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        end_points['block5'] = net

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
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3')
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3')
        end_points[end_point] = net

        # Prediction and localisations layers.
        predictions = {}
        localisations = {}
        for i, layer in enumerate(ssd_300_features):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          ssd_300_sizes[i],
                                          ssd_300_ratios[i],
                                          ssd_300_normalizations[i],
                                          clip=True, interm_layer=0)
            predictions[layer] = p
            localisations[layer] = l

        return predictions, localisations, end_points
ssd_300_vgg.default_image_size = 300
