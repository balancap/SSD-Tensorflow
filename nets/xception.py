"""Definition of Xception model introduced by F. Chollet.

Usage:
  with slim.arg_scope(xception.xception_arg_scope()):
    outputs, end_points = xception.xception(inputs)
@@xception
"""

import tensorflow as tf
slim = tf.contrib.slim


# =========================================================================== #
# Xception implementation (clean)
# =========================================================================== #
def xception(inputs,
             num_classes=1000,
             is_training=True,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             reuse=None,
             scope='xception'):
    """Xception model from https://arxiv.org/pdf/1610.02357v2.pdf

    The default image size used to train this network is 299x299.
    """

    # end_points collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}

    with tf.variable_scope(scope, 'xception', [inputs]):
        # Block 1.
        end_point = 'block1'
        with tf.variable_scope(end_point):
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID', scope='conv1')
            net = slim.conv2d(net, 64, [3, 3], padding='VALID', scope='conv2')
        end_points[end_point] = net

        # Residual block 2.
        end_point = 'block2'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 128, [1, 1], stride=2, activation_fn=None, scope='res')
            net = slim.separable_convolution2d(net, 128, [3, 3], 1, scope='sepconv1')
            net = slim.separable_convolution2d(net, 128, [3, 3], 1, activation_fn=None, scope='sepconv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        # Residual block 3.
        end_point = 'block3'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 256, [1, 1], stride=2, activation_fn=None, scope='res')
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 256, [3, 3], 1, scope='sepconv1')
            net = slim.separable_convolution2d(net, 256, [3, 3], 1, activation_fn=None, scope='sepconv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        # Residual block 4.
        end_point = 'block4'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 728, [1, 1], stride=2, activation_fn=None, scope='res')
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 728, [3, 3], 1, scope='sepconv1')
            net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None, scope='sepconv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        # Middle flow blocks.
        for i in range(8):
            end_point = 'block' + str(i + 5)
            with tf.variable_scope(end_point):
                res = net
                net = tf.nn.relu(net)
                net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                   scope='sepconv1')
                net = tf.nn.relu(net)
                net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                   scope='sepconv2')
                net = tf.nn.relu(net)
                net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                   scope='sepconv3')
                net = res + net
            end_points[end_point] = net

        # Exit flow: blocks 13 and 14.
        end_point = 'block13'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 1024, [1, 1], stride=2, activation_fn=None, scope='res')
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None, scope='sepconv1')
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 1024, [3, 3], 1, activation_fn=None, scope='sepconv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        end_point = 'block14'
        with tf.variable_scope(end_point):
            net = slim.separable_convolution2d(net, 1536, [3, 3], 1, scope='sepconv1')
            net = slim.separable_convolution2d(net, 2048, [3, 3], 1, scope='sepconv2')
        end_points[end_point] = net

        # Global averaging.
        end_point = 'dense'
        with tf.variable_scope(end_point):
            net = tf.reduce_mean(net, [1, 2], name='reduce_avg')
            logits = slim.fully_connected(net, 1000, activation_fn=None)

            end_points['logits'] = logits
            end_points['predictions'] = prediction_fn(logits, scope='Predictions')

        return logits, end_points
xception.default_image_size = 299


def xception_arg_scope(weight_decay=0.00001, stddev=0.1):
    """Defines the default Xception arg scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.

    Returns:
      An `arg_scope` to use for the xception model.
    """
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_convolution2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d, slim.separable_convolution2d],
                padding='SAME',
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as sc:
                return sc


# =========================================================================== #
# Xception arg scope (Keras hack!)
# =========================================================================== #
def xception_keras_arg_scope(hdf5_file, weight_decay=0.00001):
    """Defines an Xception arg scope which initialize layers weights
    using a Keras HDF5 file.

    Quite hacky implementaion, but seems to be working!

    Args:
      hdf5_file: HDF5 file handle.
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the xception model.
    """
    # Default batch normalization parameters.
    batch_norm_params = {
        'center': True,
        'scale': False,
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    # Read weights from HDF5 file.
    def keras_bn_params():
        def _beta_initializer(shape, dtype, partition_info=None):
            keras_bn_params.bidx += 1
            k = 'batchnormalization_%i' % keras_bn_params.bidx
            kb = 'batchnormalization_%i_beta:0' % keras_bn_params.bidx
            return tf.cast(hdf5_file[k][kb][:], dtype)

        def _gamma_initializer(shape, dtype, partition_info=None):
            keras_bn_params.gidx += 1
            k = 'batchnormalization_%i' % keras_bn_params.gidx
            kg = 'batchnormalization_%i_gamma:0' % keras_bn_params.gidx
            return tf.cast(hdf5_file[k][kg][:], dtype)

        def _mean_initializer(shape, dtype, partition_info=None):
            keras_bn_params.midx += 1
            k = 'batchnormalization_%i' % keras_bn_params.midx
            km = 'batchnormalization_%i_running_mean:0' % keras_bn_params.midx
            return tf.cast(hdf5_file[k][km][:], dtype)

        def _variance_initializer(shape, dtype, partition_info=None):
            keras_bn_params.vidx += 1
            k = 'batchnormalization_%i' % keras_bn_params.vidx
            kv = 'batchnormalization_%i_running_std:0' % keras_bn_params.vidx
            return tf.cast(hdf5_file[k][kv][:], dtype)

        # Batch normalisation initializers.
        params = batch_norm_params.copy()
        params['initializers'] = {
            'beta': _beta_initializer,
            'gamma': _gamma_initializer,
            'moving_mean': _mean_initializer,
            'moving_variance': _variance_initializer,
        }
        return params
    keras_bn_params.bidx = 0
    keras_bn_params.gidx = 0
    keras_bn_params.midx = 0
    keras_bn_params.vidx = 0

    def keras_conv2d_weights():
        def _initializer(shape, dtype, partition_info=None):
            keras_conv2d_weights.idx += 1
            k = 'convolution2d_%i' % keras_conv2d_weights.idx
            kw = 'convolution2d_%i_W:0' % keras_conv2d_weights.idx
            return tf.cast(hdf5_file[k][kw][:], dtype)
        return _initializer
    keras_conv2d_weights.idx = 0

    def keras_sep_conv2d_weights():
        def _initializer(shape, dtype, partition_info=None):
            # Depthwise or Pointwise convolution?
            if shape[0] > 1 or shape[1] > 1:
                keras_sep_conv2d_weights.didx += 1
                k = 'separableconvolution2d_%i' % keras_sep_conv2d_weights.didx
                kd = 'separableconvolution2d_%i_depthwise_kernel:0' % keras_sep_conv2d_weights.didx
                weights = hdf5_file[k][kd][:]
            else:
                keras_sep_conv2d_weights.pidx += 1
                k = 'separableconvolution2d_%i' % keras_sep_conv2d_weights.pidx
                kp = 'separableconvolution2d_%i_pointwise_kernel:0' % keras_sep_conv2d_weights.pidx
                weights = hdf5_file[k][kp][:]
            return tf.cast(weights, dtype)
        return _initializer
    keras_sep_conv2d_weights.didx = 0
    keras_sep_conv2d_weights.pidx = 0

    def keras_dense_weights():
        def _initializer(shape, dtype, partition_info=None):
            keras_dense_weights.idx += 1
            k = 'dense_%i' % keras_dense_weights.idx
            kw = 'dense_%i_W:0' % keras_dense_weights.idx
            return tf.cast(hdf5_file[k][kw][:], dtype)
        return _initializer
    keras_dense_weights.idx = 1

    def keras_dense_biases():
        def _initializer(shape, dtype, partition_info=None):
            keras_dense_biases.idx += 1
            k = 'dense_%i' % keras_dense_biases.idx
            kb = 'dense_%i_b:0' % keras_dense_biases.idx
            return tf.cast(hdf5_file[k][kb][:], dtype)
        return _initializer
    keras_dense_biases.idx = 1

    # Default network arg scope.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_convolution2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d, slim.separable_convolution2d],
                padding='SAME',
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=keras_bn_params()):
            with slim.arg_scope([slim.max_pool2d], padding='SAME'):

                # Weights initializers from Keras weights.
                with slim.arg_scope([slim.conv2d],
                                    weights_initializer=keras_conv2d_weights()):
                    with slim.arg_scope([slim.separable_convolution2d],
                                        weights_initializer=keras_sep_conv2d_weights()):
                        with slim.arg_scope([slim.fully_connected],
                                            weights_initializer=keras_dense_weights(),
                                            biases_initializer=keras_dense_biases()) as sc:
                            return sc

