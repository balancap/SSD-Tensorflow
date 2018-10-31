
# coding: utf-8

# In[28]:


import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim


# In[29]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[30]:


import sys
sys.path.append('../')


# In[31]:


from nets import ssd_vgg_300, ssd_common, np_methods,ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing
import visualization


# In[32]:


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# ## SSD 300 Model
#
# The SSD 300 network takes 300x300 image inputs. In order to feed any image, the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that even though it may change the ratio width / height, the SSD model performs well on resized images (and it is the default behaviour in the original Caffe implementation).
#
# SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors.

# In[33]:


# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
##ckpt_filename = '../checkpoints_all/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
#ckpt_filename="../checkpoints_all/VGG_VOC0712_SSD_300x300_iter_120000.ckpt/VGG_VOC0712_SSD_300x300_iter_120000.ckpt"
ckpt_filename="../logs/model.ckpt-20154"
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# ## Post-processing pipeline
#
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
#
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# In[34]:


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# In[21]:


# Test on some demo image and visualize output.
path = '../demo/'
image_names = sorted(os.listdir(path))

#img = mpimg.imread(path + image_names[-5])
cur_path="/home/hp/Pictures/1.jpg"
path="/home/hp/Pictures/1_1.jpg"
img=mpimg.imread(cur_path)
rclasses, rscores, rbboxes =  process_image(img)

visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma,path)
#visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
