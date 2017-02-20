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
"""Additional Numpy methods. Big mess of many things!
"""
import numpy as np


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




