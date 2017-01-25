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

    def load(self, filename):
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
        # Layers counters.
        self.counters['conv_weights'] = 0
        self.counters['conv_biases'] = 0

    def conv_weights_init(self):
        def _initializer(shape, dtype, partition_info=None):
            idx = self.layers['convolution'][self.counters['conv_weights']]
            layer = self.caffe_layers[idx]
            # Weights: reshape and transpose dimensions.
            w = np.array(layer.blobs[0].data)
            w = np.reshape(w, layer.blobs[0].shape.dim)
            w = np.transpose(w, (2, 3, 1, 0))
            self.counters['conv_weights'] += 1
            print('Load weights from convolution layer:', layer.name, w.shape)
            return tf.cast(w, dtype)
        return _initializer

    def conv_biases_init(self):
        def _initializer(shape, dtype, partition_info=None):
            idx = self.layers['convolution'][self.counters['conv_biases']]
            layer = self.caffe_layers[idx]
            # Weights: reshape and transpose dimensions.
            b = np.array(layer.blobs[1].data)
            self.counters['conv_biases'] += 1
            print('Load biases from convolution layer:', layer.name, b.shape)
            return tf.cast(b, dtype)
        return _initializer
