"""Pre-processing images for SSD-type networks.
"""
from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

# Resizing strategies.
Resize = Enum('Resize', ('NONE',                # Nothing!
                         'CENTRAL_CROP',        # Crop (and pad if necessary).
                         'PAD_AND_RESIZE'))     # Pad, and resize to output .shape

# VGG mean parameters.
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(2, num_channels, image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(2, channels)


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(2, num_channels, image)
    for i in range(num_channels):
        channels[i] += means[i]
    image = tf.concat(2, channels)
    if to_int:
        image = tf.cast(image, tf.int32)
    return image


def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    img = np.copy(image)
    img[:, :, 0] += means[0]
    img[:, :, 1] += means[1]
    img[:, :, 2] += means[2]
    if to_int:
        img = img.astype(np.uint8)
    return img


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.

    Returns:
        A preprocessed image.
    """
    # resize_side = tf.random_uniform(
    #         [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

    # image = _aspect_preserving_resize(image, resize_side)
    # image = _random_crop([image], output_height, output_width)[0]
    # image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    return tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_for_eval(image, out_shape, resize):
    """Preprocess an image for evaluation.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        out_shape: Output shape after pre-processing (if resize != None)
        resize: Resize strategy.

    Returns:
        A preprocessed image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    image = tf.to_float(image)
    image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    # Resize strategy...
    if resize == Resize.NONE:
        pass
    elif resize == Resize.CENTRAL_CROP:
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       out_shape[0],
                                                       out_shape[1])
    elif resize == Resize.PAD_AND_RESIZE:
        # Resizing first: find the correct factor...
        shape = tf.shape(image)
        factor = tf.minimum(tf.to_double(1.0),
                            tf.minimum(tf.to_double(out_shape[0] / shape[0]),
                                       tf.to_double(out_shape[1] / shape[1])))
        # Resize image.
        resize_shape = tf.stack([factor * tf.to_double(shape[0]),
                                 factor * tf.to_double(shape[1])])
        resize_shape = tf.cast(tf.floor(resize_shape), tf.int32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image,
                                       resize_shape,
                                       method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=False)
        image = tf.squeeze(image)
        # image = tf.reshape(image, tf.concat(0, [resize_shape, [3]]))
        # Pad to expected size.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       out_shape[0],
                                                       out_shape[1])
        image = tf.reshape(image, [*out_shape, 3])
    return image


def preprocess_image(image,
                     out_shape,
                     is_training=False,
                     resize=Resize.CENTRAL_CROP):
    """Pre-process an given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
         ignored. Otherwise, the resize side is sampled from
         [resize_size_min, resize_size_max].

    Returns:
        A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, out_shape, resize)
    else:
        return preprocess_for_eval(image, out_shape, resize)
