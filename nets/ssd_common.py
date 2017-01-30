"""Shared function between different SSD implementations.
"""
import numpy as np
import tensorflow as tf


# =========================================================================== #
# TensorFlow implementation of some bboxes methods.
# =========================================================================== #
def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         matching_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32):
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors
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
    feat_ymax = tf.zeros(shape, dtype=dtype)
    feat_xmax = tf.zeros(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score a box and the anchors.
        """
        # Intersection bbox and volume.
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        # Volumes.
        inter_vol = (int_xmax - int_xmin) * (int_ymax - int_ymin)
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        jaccard = inter_vol / union_vol
        return jaccard

    def condition(i,
                  feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i,
             feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        scores = jaccard_with_anchors(bbox)
        # 'Boolean' mask.
        mask = tf.greater(scores, matching_threshold) & \
            tf.greater(scores, feat_scores)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = fmask * scores + (1 - fmask) * feat_scores

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax
        return [i+1,
                feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i,
     feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition,
                                           body,
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


def tf_ssd_bboxes_decode(feat_localizations,
                         anchor_bboxes,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchor_bboxes: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchor_bboxes
    # yref = np.expand_dims(yref, axis=-1)
    # xref = np.expand_dims(xref, axis=-1)

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


# =========================================================================== #
# Numpy implementations of common boxes functions.
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


def ssd_bboxes_from_features(feat_predictions,
                             feat_localizations,
                             anchor_bboxes,
                             threshold=0.5,
                             img_shape=(300, 300),
                             num_classes=21):
    """Extract classes, scores and bounding boxes from features in one layer.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    # Reshape features: N x N_Anchors x N_labels|4
    shape = feat_predictions.shape
    feat_predictions = np.reshape(feat_predictions, (np.prod(shape[:-2]), shape[-2], shape[-1]))
    shape = feat_localizations.shape
    feat_localizations = np.reshape(feat_localizations, (np.prod(shape[:-2]), shape[-2], shape[-1]))

    # Predictions, removing first void class.
    sub_predictions = feat_predictions[:, :, 1:]
    idxes = np.where(sub_predictions > threshold)
    classes = idxes[-1]+1
    scores = sub_predictions[idxes]

    # Decode localizations features and get bboxes.
    bboxes = feat_localizations
    bboxes = ssd_bboxes_decode(feat_localizations, anchor_bboxes)
    bboxes = bboxes[idxes[:-1]]

    return classes, scores, bboxes


def ssd_bboxes_from_layers(layers_predictions,
                           layers_localizations,
                           layers_anchors,
                           threshold=0.5,
                           img_shape=(300, 300),
                           num_classes=21):
    """Extract classes, scores and bounding boxes from network output layers.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    for i in range(len(layers_predictions)):
        feat_predictions = layers_predictions[i]
        feat_localizations = layers_localizations[i]
        anchor_bboxes = layers_anchors[i]

        classes, scores, bboxes = ssd_bboxes_from_features(feat_predictions,
                                                           feat_localizations,
                                                           anchor_bboxes,
                                                           threshold,
                                                           img_shape,
                                                           num_classes)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)

        # Some debug info?
#         print('Features', i, "shape :", feat_predictions.shape, feat_localizations.shape)
#         print('Classes', classes)
#         print('Scores:', scores)
#         print('bboxes:', bboxes.shape)
#         print('')
    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes


def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 can be multi-dimensional.
    """
    if bboxes1.ndim == 1:
        bboxes1 = np.expand_dims(bboxes1, 0)
    if bboxes2.ndim == 1:
        bboxes2 = np.expand_dims(bboxes2, 0)
    # Intersection bbox and volume.
    int_bbox = np.vstack([np.maximum(bboxes1[:, 0], bboxes2[:, 0]),
                          np.maximum(bboxes1[:, 1], bboxes2[:, 1]),
                          np.minimum(bboxes1[:, 2], bboxes2[:, 2]),
                          np.minimum(bboxes1[:, 3], bboxes2[:, 3])])
    int_bbox = np.transpose(int_bbox)
    int_vol = (int_bbox[:, 2] - int_bbox[:, 0]) * (int_bbox[:, 3] - int_bbox[:, 1])
    vol1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    vol2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard


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


