"""Specific Caffe scope used to import weights from a .caffemodel file.

The idea is to create special initializers loading weights from protobuf
.caffemodel files.
"""
import caffe
from caffe.proto import caffe_pb2

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


class CaffeScope(object):
    """Caffe scope.
    """
    def __init__(self):
        """Initialize the caffee scope.
        """
        self.counters = {}
        self.layers = {}
        self.caffe_layers = None
        self.bgr_to_rgb = 0

    def load(self, filename, bgr_to_rgb=True):
        """Load weights from a .caffemodel file and initialize counters.

        Params:
          filename: caffemodel file.
        """
        print('Loading Caffe file:', filename)
        caffemodel_params = caffe_pb2.NetParameter()
        caffemodel_str = open(filename, 'rb').read()
        caffemodel_params.ParseFromString(caffemodel_str)
        self.caffe_layers = caffemodel_params.layer

        # Layers collection.
        self.layers['convolution'] = [i for i, l in enumerate(self.caffe_layers)
                                      if l.type == 'Convolution']
        self.layers['l2_normalization'] = [i for i, l in enumerate(self.caffe_layers)
                                           if l.type == 'Normalize']
        # BGR to RGB convertion. Tries to find the first convolution with 3
        # and exchange parameters.
        if bgr_to_rgb:
            self.bgr_to_rgb = 1

    def conv_weights_init(self):
        def _initializer(shape, dtype, partition_info=None):
            counter = self.counters.get(self.conv_weights_init, 0)
            idx = self.layers['convolution'][counter]
            layer = self.caffe_layers[idx]
            # Weights: reshape and transpose dimensions.
            w = np.array(layer.blobs[0].data)
            w = np.reshape(w, layer.blobs[0].shape.dim)
            # w = np.transpose(w, (1, 0, 2, 3))
            w = np.transpose(w, (2, 3, 1, 0))
            if self.bgr_to_rgb == 1 and w.shape[2] == 3:
                print('Convert BGR to RGB in convolution layer:', layer.name)
                w[:, :, (0, 1, 2)] = w[:, :, (2, 1, 0)]
                self.bgr_to_rgb += 1
            self.counters[self.conv_weights_init] = counter + 1
            print('Load weights from convolution layer:', layer.name, w.shape)
            return tf.cast(w, dtype)
        return _initializer

    def conv_biases_init(self):
        def _initializer(shape, dtype, partition_info=None):
            counter = self.counters.get(self.conv_biases_init, 0)
            idx = self.layers['convolution'][counter]
            layer = self.caffe_layers[idx]
            # Biases data...
            b = np.array(layer.blobs[1].data)
            self.counters[self.conv_biases_init] = counter + 1
            print('Load biases from convolution layer:', layer.name, b.shape)
            return tf.cast(b, dtype)
        return _initializer

    def l2_norm_scale_init(self):
        def _initializer(shape, dtype, partition_info=None):
            counter = self.counters.get(self.l2_norm_scale_init, 0)
            idx = self.layers['l2_normalization'][counter]
            layer = self.caffe_layers[idx]
            # Scaling parameter.
            s = np.array(layer.blobs[0].data)
            s = np.reshape(s, layer.blobs[0].shape.dim)
            self.counters[self.l2_norm_scale_init] = counter + 1
            print('Load scaling from L2 normalization layer:', layer.name, s.shape)
            return tf.cast(s, dtype)
        return _initializer
