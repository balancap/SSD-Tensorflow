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
"""Shared function between different SSD implementations.
"""
import numpy as np
import tensorflow as tf


# =========================================================================== #
# TensorFlow implementation of boxes SSD encoding / decoding.
# =========================================================================== #
def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...
        interscts = intersection_with_anchors(bbox)
        mask = tf.logical_and(interscts > ignore_threshold,
                              label == no_annotation_label)
        # Replace scores by -1.
        feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores


def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores = \
                    tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes, no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, dtype)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
        return target_labels, target_localizations, target_scores


def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes


def tf_ssd_bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_ssd_bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes


def tf_ssd_bboxes_select_layer(predictions_layer,
                               localizations_layer):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Return:
      classes, scores, bboxes: Input Tensors.
    """
    # Reshape features: Batches x N x N_labels | 4
    p_shape = tf_shape(predictions_layer)
    predictions_layer = tf.reshape(predictions_layer,
                                   tf.stack([p_shape[0], -1, p_shape[-1]]))
    l_shape = tf_shape(localizations_layer)
    localizations_layer = tf.reshape(localizations_layer,
                                     tf.stack([l_shape[0], -1, l_shape[-1]]))
    # Class prediction and scores: assign 0. to 0-class
    classes = tf.argmax(predictions_layer, axis=2)
    scores = tf.reduce_max(predictions_layer, axis=2)
    scores = scores * tf.cast(classes > 0, scores.dtype)

    bboxes = localizations_layer

    return classes, scores, bboxes


def tf_ssd_bboxes_select(predictions_net,
                         localizations_net,
                         scope=None):
    """Extract classes, scores and bounding boxes from network output layers.

    Return:
      classes, scores, bboxes: Tensors.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_classes = []
        l_scores = []
        l_bboxes = []
        # l_layers = []
        # l_idxes = []
        for i in range(len(predictions_net)):
            classes, scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
                                                                 localizations_net[i])
            l_classes.append(classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)

        classes = tf.concat(l_classes, axis=1)
        scores = tf.concat(l_scores, axis=1)
        bboxes = tf.concat(l_bboxes, axis=1)
        return classes, scores, bboxes


# =========================================================================== #
# Additional TF bboxes methods.
# =========================================================================== #
def tf_shape(x, rank=None):
    """Returns the dimensions of an image tensor.
    Args:
      image: A 3-D Tensor of shape `[height, width, channels]`.
    Returns:
      A list of `[height, width, channels]` corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def tf_pad_axis(x, offset, size, axis=0, name=None):
    with tf.name_scope(name, 'pad_axis'):
        shape = tf_shape(x)
        rank = len(shape)
        pad1 = tf.stack([0]*axis + [offset] + [0]*(rank-axis-1))
        pad2 = tf.stack([0]*axis + [tf.maximum(size-offset-shape[axis], 0)] + [0]*(rank-axis-1))
        paddings = tf.stack([pad1, pad2], axis=1)
        x = tf.pad(x, paddings, mode='CONSTANT')
        # Reshape, to get fully defined shape if possible.
        shape[axis] = size
        x = tf.reshape(x, tf.stack(shape))
        return x


def tf_bboxes_sort(classes, scores, bboxes, top_k=400, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    Batch-compatible.
    """
    with tf.name_scope(scope, 'bboxes_sort', [classes, scores, bboxes]):
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the batch.
        def fn_gather(classes, bboxes, idxes):
            cl = tf.gather(classes, idxes)
            bb = tf.gather(bboxes, idxes)
            return [cl, bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1], x[2]),
                      [classes, bboxes, idxes],
                      dtype=[classes.dtype, bboxes.dtype],
                      parallel_iterations=5,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        classes = r[0]
        bboxes = r[1]
        return classes, scores, bboxes


def tf_bboxes_clip(bbox_ref, bboxes, scope=None):
    """Clip bounding boxes to a reference box.
    Batch-compatible.
    """
    with tf.name_scope(scope, 'bboxes_clip'):
        bboxes = tf.maximum(bboxes, tf.stack([bbox_ref[0], bbox_ref[1],
                                              -np.inf, -np.inf]))
        bboxes = tf.minimum(bboxes, tf.stack([np.inf, np.inf,
                                              bbox_ref[2], bbox_ref[3]]))
        return bboxes


def tf_bboxes_nms(classes, scores, bboxes,
                  nms_threshold=0.5, max_objects=50,
                  num_classes=21, pad=True, scope=None):
    """Apply non-maximum selection to bounding boxes.
    Should only be used on single-entries. Use batch version otherwise.
    """
    with tf.name_scope(scope, 'bboxes_nms_single'):
        max_output_size = tf_shape(classes)[-1]
        l_classes = []
        l_scores = []
        l_bboxes = []
        # Apply NMS algorithm on every class.
        for i in range(1, num_classes):
            mask = tf.equal(classes, i)
            sub_scores = tf.boolean_mask(scores, mask)
            sub_bboxes = tf.boolean_mask(bboxes, mask)
            sub_classes = tf.boolean_mask(classes, mask)
            idxes = tf.image.non_max_suppression(sub_bboxes, sub_scores,
                                                 max_output_size, nms_threshold)
            l_classes.append(tf.gather(sub_classes, idxes))
            l_scores.append(tf.gather(sub_scores, idxes))
            l_bboxes.append(tf.gather(sub_bboxes, idxes))
        # Concat results.
        classes = tf.concat(tf.tuple(l_classes), axis=0)
        scores = tf.concat(tf.tuple(l_scores), axis=0)
        bboxes = tf.concat(tf.tuple(l_bboxes), axis=0)
        # Sort by score.
        scores, idxes = tf.nn.top_k(scores, k=tf.size(scores), sorted=True)
        classes = tf.gather(classes, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad outputs to initial size. Necessary if use for in batches...
        if pad:
            classes = tf_pad_axis(classes, 0, max_objects, axis=0)
            scores = tf_pad_axis(scores, 0, max_objects, axis=0)
            bboxes = tf_pad_axis(bboxes, 0, max_objects, axis=0)
        return classes, scores, bboxes


def tf_bboxes_nms_batch(classes, scores, bboxes,
                        nms_threshold=0.5, max_objects=50,
                        num_classes=21, scope=None):
    """Apply non-maximum selection to bounding boxes.
    Suitable for batched inputs. In order to have the same output shape for
    all elements of the batch, we use zero-padding.
    """
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: tf_bboxes_nms(x[0], x[1], x[2],
                                              nms_threshold, max_objects,
                                              num_classes, pad=True),
                      (classes, scores, bboxes),
                      dtype=None,
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        return r[0], r[1], r[2]


def _tf_select_index(idx, val, t):
    """Return a tensor.
    """
    idx = tf.expand_dims(tf.expand_dims(idx, 0), 0)
    val = tf.expand_dims(val, 0)
    t = t + tf.scatter_nd(idx, val, tf.shape(t))
    return t


def tf_bboxes_matching(rclasses, rscores, rbboxes, glabels, gbboxes,
                       matching_threshold=0.5, scope=None):
    with tf.name_scope(scope, 'bboxes_matching_single',
                       [rclasses, rscores, rbboxes, glabels, gbboxes]):
        rsize = tf.size(rclasses)
        rshape = tf.shape(rclasses)
        # Number of groundtruth boxes.
        n_gbboxes = tf.count_nonzero(glabels)
        # Grountruth matching arrays.
        gmatch = tf.zeros(tf.shape(glabels), dtype=tf.bool)
        grange = tf.range(tf.size(glabels), dtype=tf.int32)

        # Matching TensorArrays.
        ta_tp_bool = tf.TensorArray(tf.bool, size=rsize,
                                    dynamic_size=False, infer_shape=True)
        ta_fp_bool = tf.TensorArray(tf.bool, size=rsize,
                                    dynamic_size=False, infer_shape=True)

        # Loop over returned objects.
        def m_condition(i, ta_tp, ta_fp, gmatch):
            r = tf.less(i, rsize)
            return r

        def m_body(i, ta_tp, ta_fp, gmatch):
            """Update matching scores: check same class and the score is greater.
            """
            # Jaccard score with groundtruth bboxes.
            rbbox = rbboxes[i]
            rlabel = rclasses[i]
            jaccard = tf_bboxes_jaccard(rbbox, gbboxes)
            # jaccard = jaccard * tf.cast(glabels == rlabel, dtype=jaccard.dtype)

            # Best fit, checking it's above threshold.
            idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)
            jcdmax = jaccard[idxmax]
            match = jcdmax > matching_threshold
            existing_match = gmatch[idxmax]
            # TP: match & no previous match and FP: previous match | no match.
            ta_tp = ta_tp.write(i, tf.logical_and(match, tf.logical_not(existing_match)))
            ta_fp = ta_fp.write(i, tf.logical_or(existing_match, tf.logical_not(match)))
            # Update grountruth match.
            mask = tf.logical_and(tf.equal(grange, idxmax), match)
            gmatch = tf.logical_or(gmatch, mask)

            return [i+1, ta_tp, ta_fp, gmatch]
        # Main loop definition.
        i = 0
        [i, ta_tp_bool, ta_fp_bool, gmatch] = \
            tf.while_loop(m_condition, m_body,
                          [i, ta_tp_bool, ta_fp_bool, gmatch],
                          back_prop=False)
        # TensorArrays to Tensors and reshape.
        tp_match = tf.reshape(ta_fp_bool.stack(), rshape)
        fp_match = tf.reshape(ta_tp_bool.stack(), rshape)
        return n_gbboxes, tp_match, fp_match


def tf_bboxes_matching_batch(rclasses, rscores, rbboxes, glabels, gbboxes,
                             matching_threshold=0.5, scope=None):
    with tf.name_scope(scope, 'bboxes_matching_single',
                       [rclasses, rscores, rbboxes, glabels, gbboxes]):
        r = tf.map_fn(lambda x: tf_bboxes_matching(x[0], x[1], x[2], x[3], x[4],
                                                   matching_threshold),
                      (rclasses, rscores, rbboxes, glabels, gbboxes),
                      dtype=(tf.int64, tf.bool, tf.bool),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=True,
                      infer_shape=True)
        return r[0], r[1], r[2]


def tf_bboxes_jaccard(bbox_ref, bboxes, name=None):
    """Compute jaccard score between a reference box and a collection
    of bounding boxes.

    Args:
      bbox_ref: Reference bounding box. 1x4 or 4 Tensor.
      bboxes: Nx4 Tensor, collection of bounding boxes.
    Return:
      Nx4 Tensor with Jaccard scores.
    """
    with tf.name_scope(name, 'bboxes_jaccard'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = -inter_vol \
            + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
            + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
        jaccard = tf.divide(inter_vol, union_vol)
        return jaccard


def tf_bboxes_intersection(bbox_ref, bboxes, name=None):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.

    Args:
      bbox_ref: Reference bounding box. 1x4 or 4 Tensor.
      bboxes: Nx4 Tensor, collection of bounding boxes.
    Return:
      Nx4 Tensor with relative intersection.
    """
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = tf.div(inter_vol, bboxes_vol)
        return scores


def tf_bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Resize.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


def tf_bboxes_filter_center(labels, bboxes,
                            margins=[0., 0., 0., 0.],
                            scope=None):
    """Filter out bounding boxes whose center are not in
    the rectangle [0, 0, 1, 1] + margins. The margin Tensor
    can be used to enforce or loosen this condition.

    Return:
      labels, bboxes: Filtered elements.
    """
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        cy = (bboxes[:, 0] + bboxes[:, 2]) / 2.
        cx = (bboxes[:, 1] + bboxes[:, 3]) / 2.
        mask = tf.greater(cy, margins[0])
        mask = tf.logical_and(mask, tf.greater(cx, margins[1]))
        mask = tf.logical_and(mask, tf.less(cx, 1. + margins[2]))
        mask = tf.logical_and(mask, tf.less(cx, 1. + margins[3]))
        # Boolean masking...
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


def tf_bboxes_filter_overlap(labels, bboxes,
                             threshold=0.5,
                             scope=None):
    """Filter out bounding boxes based on overlap with reference
    box [0, 0, 1, 1].

    Return:
      labels, bboxes: Filtered elements.
    """
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = tf_bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),
                                        bboxes)
        # Boolean masking...
        mask = scores > threshold
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


def tf_bboxes_filter_labels(labels, bboxes,
                            out_labels=[], num_classes=np.inf,
                            scope=None):
    """Filter out labels from a collection. Typically used to get
    of DontCare elements. Also remove elements based on the number of classes.

    Return:
      labels, bboxes: Filtered elements.
    """
    with tf.name_scope(scope, 'bboxes_filter_labels', [labels, bboxes]):
        mask = tf.greater_equal(labels, num_classes)
        for l in labels:
            mask = tf.logical_and(mask, tf.not_equal(labels, l))
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


# =========================================================================== #
# Numpy implementations of SSD boxes functions.
# =========================================================================== #
def ssd_bboxes_decode(feat_localizations,
                      anchor_bboxes,
                      prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Return:
      numpy array Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchor_bboxes
    xref = np.reshape(xref, [np.prod(xref.shape), 1])
    yref = np.reshape(yref, [np.prod(yref.shape), 1])

    # Compute center, height and width
    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
    w = wref * np.exp(feat_localizations[:, :, 2] * prior_scaling[2])
    h = href * np.exp(feat_localizations[:, :, 3] * prior_scaling[3])
    # bboxes: ymin, xmin, xmax, ymax.
    bboxes = np.zeros_like(feat_localizations)
    bboxes[:, :, 0] = cy - h / 2.
    bboxes[:, :, 1] = cx - w / 2.
    bboxes[:, :, 2] = cy + h / 2.
    bboxes[:, :, 3] = cx + w / 2.
    return bboxes


def ssd_bboxes_select_layer(predictions_layer,
                            localizations_layer,
                            anchors_layer,
                            threshold=0.5,
                            img_shape=(300, 300),
                            num_classes=21,
                            decode=True):
    """Extract classes, scores and bounding boxes from features in one layer.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    # Reshape features: N x N_Anchors x N_labels|4
    shape = predictions_layer.shape
    predictions_layer = np.reshape(predictions_layer,
                                   (np.prod(shape[:-2]), shape[-2], shape[-1]))
    shape = localizations_layer.shape
    localizations_layer = np.reshape(localizations_layer,
                                     (np.prod(shape[:-2]), shape[-2], shape[-1]))

    # Predictions, removing first void class.
    sub_predictions = predictions_layer[:, :, 1:]
    idxes = np.where(sub_predictions > threshold)
    classes = idxes[-1]+1
    scores = sub_predictions[idxes]
    # Decode localizations features and get bboxes.
    bboxes = localizations_layer
    if decode:
        bboxes = ssd_bboxes_decode(localizations_layer, anchors_layer)
    bboxes = bboxes[idxes[:-1]]

    return classes, scores, bboxes, idxes[:-1]


def ssd_bboxes_select(predictions_net,
                      localizations_net,
                      anchors_net,
                      threshold=0.5,
                      img_shape=(300, 300),
                      num_classes=21,
                      decode=True):
    """Extract classes, scores and bounding boxes from network output layers.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    l_layers = []
    l_idxes = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes, idxes = ssd_bboxes_select_layer(
            predictions_net[i], localizations_net[i], anchors_net[i],
            threshold, img_shape, num_classes, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)
        # Debug information.
        l_layers.append(i)
        l_idxes.append((i, idxes))

    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    # layers = np.concatenate(l_layers, 0)
    return classes, scores, bboxes, l_layers, l_idxes


# =========================================================================== #
# Common functions for bboxes handling and selection.
# =========================================================================== #
def bboxes_sort(classes, scores, bboxes,
                top_k=400, priority_inside=True, margin=0.05):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    if priority_inside:
        inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
            (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
        idxes = np.argsort(-scores)
        inside = inside[idxes]
        idxes = np.concatenate([idxes[inside], idxes[~inside]])
    else:
        idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_clip(bbox_ref, bboxes):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    bboxes[:, 0] = np.maximum(bboxes[:, 0], bbox_ref[0])
    bboxes[:, 1] = np.maximum(bboxes[:, 1], bbox_ref[1])
    bboxes[:, 2] = np.minimum(bboxes[:, 2], bbox_ref[2])
    bboxes[:, 3] = np.minimum(bboxes[:, 3], bbox_ref[3])
    return bboxes


def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    bboxes = np.copy(bboxes)
    # Translate.
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    # Resize.
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes


def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 can be multi-dimensional.
    """
    if bboxes1.ndim == 1:
        bboxes1 = np.expand_dims(bboxes1, 0)
    if bboxes2.ndim == 1:
        bboxes2 = np.expand_dims(bboxes2, 0)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    int_xmin = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    int_ymax = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
    int_xmax = np.minimum(bboxes1[:, 3], bboxes2[:, 3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    vol2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard


def bboxes_intersection(bbox, bboxes):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 can be multi-dimensional.
    """
    if bboxes.ndim == 1:
        bboxes = np.expand_dims(bboxes, 0)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes[:, 0], bbox[0])
    int_xmin = np.maximum(bboxes[:, 1], bbox[1])
    int_ymax = np.minimum(bboxes[:, 2], bbox[2])
    int_xmax = np.minimum(bboxes[:, 3], bbox[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    score = int_vol / vol
    return score


def bboxes_nms(classes, scores, bboxes, threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


def bboxes_nms_fast(classes, scores, bboxes, threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    pass


