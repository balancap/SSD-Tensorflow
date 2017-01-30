"""Shared function between different SSD implementations.
"""
import numpy as np
import tensorflow as tf


def ssd_bboxes_decode(features, anchor_bboxes):
    """Compute the bounding boxes from the layer features and reference
    anchor bounding boxes.

    Return:
      numpy array Nx4: ymin, xmin, ymax, xmax
    """
    prior_scaling = [0.1, 0.1, 0.2, 0.2]
    yref, xref, href, wref = anchor_bboxes
    xref = np.reshape(xref, [np.prod(xref.shape), 1])
    yref = np.reshape(yref, [np.prod(yref.shape), 1])

    # Compute center, height and width
    cx = features[:, :, 0] * wref * prior_scaling[0] + xref
    cy = features[:, :, 1] * href * prior_scaling[1] + yref
    w = wref * np.exp(features[:, :, 2] * prior_scaling[2])
    h = href * np.exp(features[:, :, 3] * prior_scaling[3])
    # bboxes: ymin, xmin, xmax, ymax.
    bboxes = np.zeros_like(features)
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


