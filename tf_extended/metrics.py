# Copyright 2017 Paul Balanca. All Rights Reserved.
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
"""TF Extended: additional metrics.
"""
import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import tensor_util
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.metrics.python.ops import set_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


# =========================================================================== #
# TensorFlow utils
# =========================================================================== #
def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=dtypes.float32):
    """Creates a new local variable.
    Args:
        name: The name of the new or existing variable.
        shape: Shape of the new or existing variable.
        collections: A list of collection names to which the Variable will be added.
        validate_shape: Whether to validate the shape of the variable.
        dtype: Data type of the variables.
    Returns:
        The created variable.
    """
    # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
    collections = list(collections or [])
    collections += [ops.GraphKeys.LOCAL_VARIABLES]
    return variables.Variable(
            initial_value=array_ops.zeros(shape, dtype=dtype),
            name=name,
            trainable=False,
            collections=collections,
            validate_shape=validate_shape)


def _safe_div(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return math_ops.select(
        math_ops.greater(denominator, 0),
        math_ops.div(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)


def _broadcast_weights(weights, values):
    """Broadcast `weights` to the same shape as `values`.
    This returns a version of `weights` following the same broadcast rules as
    `mul(weights, values)`. When computing a weighted average, use this function
    to broadcast `weights` before summing them; e.g.,
    `reduce_sum(w * v) / reduce_sum(_broadcast_weights(w, v))`.
    Args:
      weights: `Tensor` whose shape is broadcastable to `values`.
      values: `Tensor` of any shape.
    Returns:
      `weights` broadcast to `values` shape.
    """
    weights_shape = weights.get_shape()
    values_shape = values.get_shape()
    if(weights_shape.is_fully_defined() and
       values_shape.is_fully_defined() and
       weights_shape.is_compatible_with(values_shape)):
        return weights
    return math_ops.mul(
        weights, array_ops.ones_like(values), name='broadcast_weights')


def _precision_recall(n_gbboxes, scores, tp, fp, scope=None):
    """Compute precision and recall from scores, true positives and false
    positives.
    """
    # Sort by score.
    with tf.name_scope(scope, 'prec_rec'):
        scores, idxes = tf.nn.top_k(scores, k=tf.size(scores), sorted=True)
        tp = tf.gather(tp, idxes)
        fp = tf.gather(fp, idxes)
        # Computer recall and precision.
        tp = tf.cumsum(tp, axis=0)
        fp = tf.cumsum(fp, axis=0)
        recall = _safe_div(tp, tf.cast(n_gbboxes, tp.dtype), 'recall')
        precision = _safe_div(tp, tp + fp, 'precision')
        return tf.tuple([scores, precision, recall])


# =========================================================================== #
# TF Extended metrics
# =========================================================================== #
def streaming_precision_recall_arrays(n_gbboxes, rclasses, rscores,
                                      tp_tensor, fp_tensor,
                                      remove_zero_labels=True,
                                      metrics_collections=None,
                                      updates_collections=None, scope=None):

    with variable_scope.variable_scope(scope, 'precision_recall',
                                       [n_gbboxes, rclasses, tp_tensor, fp_tensor]):
        n_gbboxes = math_ops.to_int64(n_gbboxes)
        rclasses = math_ops.to_int64(rclasses)
        rscores = math_ops.to_float(rscores)
        tp_tensor = math_ops.to_float(tp_tensor)
        fp_tensor = math_ops.to_float(fp_tensor)
        # Reshape TP and FP tensors and clean away 0 classes.
        rclasses = tf.reshape(rclasses, [-1])
        rscores = tf.reshape(rscores, [-1])
        tp_tensor = tf.reshape(tp_tensor, [-1])
        fp_tensor = tf.reshape(tp_tensor, [-1])
        if remove_zero_labels:
            mask = tf.greater(rclasses, 0)
            rclasses = tf.boolean_mask(rclasses, mask)
            rscores = tf.boolean_mask(rscores, mask)
            tp_tensor = tf.boolean_mask(tp_tensor, mask)
            fp_tensor = tf.boolean_mask(fp_tensor, mask)

        # Local variables accumlating information over batches.
        v_nobjects = _create_local('v_tp', shape=[], dtype=tf.int64)
        v_scores = _create_local('v_scores', shape=[0, ])
        v_tp = _create_local('v_tp', shape=[0, ])
        v_fp = _create_local('v_fp', shape=[0, ])

        # Update operations.
        nobjects_op = state_ops.assign_add(v_nobjects, tf.reduce_sum(n_gbboxes))
        scores_op = state_ops.assign(v_scores, tf.concat(0, [v_scores, rscores]))
        tp_op = state_ops.assign(v_tp, tf.concat(0, [v_tp, tp_tensor]))
        fp_op = state_ops.assign(v_fp, tf.concat(0, [v_fp, fp_tensor]))

        # Precision and recall computations.
        r = _precision_recall(v_nobjects, v_scores, v_tp, v_fp, 'value')
        with ops.control_dependencies([nobjects_op, scores_op, tp_op, fp_op]):
            update_op = _precision_recall(v_nobjects, v_scores, v_tp, v_fp,
                                          'update_op')

        if metrics_collections:
            ops.add_to_collections(metrics_collections, r)
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return r, update_op


def streaming_mean(values, weights=None, metrics_collections=None,
                   updates_collections=None, name=None):
    """Computes the (weighted) mean of the given values.
    The `streaming_mean` function creates two local variables, `total` and `count`
    that are used to compute the average of `values`. This average is ultimately
    returned as `mean` which is an idempotent operation that simply divides
    `total` by `count`.
    For estimation of the metric  over a stream of data, the function creates an
    `update_op` operation that updates these variables and returns the `mean`.
    `update_op` increments `total` with the reduced sum of the product of `values`
    and `weights`, and it increments `count` with the reduced sum of `weights`.
    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
    Args:
        values: A `Tensor` of arbitrary dimensions.
        weights: An optional `Tensor` whose shape is broadcastable to `values`.
        metrics_collections: An optional list of collections that `mean`
            should be added to.
        updates_collections: An optional list of collections that `update_op`
            should be added to.
        name: An optional variable_scope name.
    Returns:
        mean: A tensor representing the current mean, the value of `total` divided
            by `count`.
        update_op: An operation that increments the `total` and `count` variables
            appropriately and whose value matches `mean_value`.
    Raises:
        ValueError: If `weights` is not `None` and its shape doesn't match `values`,
            or if either `metrics_collections` or `updates_collections` are not a list
            or tuple.
    """
    with variable_scope.variable_scope(name, 'mean', [values, weights]):
        values = math_ops.to_float(values)

        total = _create_local('total', shape=[])
        count = _create_local('count', shape=[])

        if weights is not None:
            weights = math_ops.to_float(weights)
            values = math_ops.mul(values, weights)
            num_values = math_ops.reduce_sum(_broadcast_weights(weights, values))
        else:
            num_values = math_ops.to_float(array_ops.size(values))

        total_compute_op = state_ops.assign_add(total, math_ops.reduce_sum(values))
        count_compute_op = state_ops.assign_add(count, num_values)

        mean = _safe_div(total, count, 'value')
        with ops.control_dependencies([total_compute_op, count_compute_op]):
            update_op = _safe_div(total, count, 'update_op')

        if metrics_collections:
            ops.add_to_collections(metrics_collections, mean)

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return mean, update_op
