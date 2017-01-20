"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import numpy as np
import tensorflow as tf

VOC_LABELS = {
    'person': (1, 'Person'),
    'bird': (2, 'Animal'),
    'cat': (3, 'Animal'),
    'cow': (4, 'Animal'),
    'dog': (5, 'Animal'),
    'horse': (6, 'Animal'),
    'sheep': (7, 'Animal'),
    'aeroplane': (8, 'Vehicle'),
    'bicycle': (9, 'Vehicle'),
    'boat': (10, 'Vehicle'),
    'bus': (11, 'Vehicle'),
    'car': (12, 'Vehicle'),
    'motorbike': (13, 'Vehicle'),
    'train': (14, 'Vehicle'),
    'bottle': (15, 'Indoor'),
    'chair': (16, 'Indoor'),
    'diningtable': (17, 'Indoor'),
    'pottedplant': (18, 'Indoor'),
    'sofa': (19, 'Indoor'),
    'tvmonitor': (20, 'Indoor'),
}
