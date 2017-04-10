# =========================================================================== #
# Dataset convert...
# =========================================================================== #
rm events* graph* model* checkpoint
mv events* graph* model* checkpoint ./log

DATASET_DIR=/media/paul/DataExt4/PascalVOC/rawdata/VOC2012/trainval/
OUTPUT_DIR=/media/paul/DataExt4/PascalVOC/dataset
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2012_train \
    --output_dir=${OUTPUT_DIR}

CAFFE_MODEL=/media/paul/DataExt4/PascalVOC/training/ckpts/SSD_300x300_VOC0712/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
python caffe_to_tensorflow.py \
    --model_name=ssd_300_vgg \
    --num_classes=21 \
    --caffemodel_path=${CAFFE_MODEL}

# =========================================================================== #
# VGG-based SSD network
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/PascalVOC/dataset
TRAIN_DIR=./logs/ssd_300_vgg_3
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2012 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.95 \
    --batch_size=32

DATASET_DIR=/media/paul/DataExt4/PascalVOC/dataset
TRAIN_DIR=./logs/ssd_300_vgg_3
EVAL_DIR=${TRAIN_DIR}/eval
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500


DATASET_DIR=/media/paul/DataExt4/PascalVOC/dataset
EVAL_DIR=./logs/ssd_300_vgg_1_eval
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_300x300_iter_120000.ckpt
CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1 \
    --max_num_batches=10


DATASET_DIR=/media/paul/DataExt4/PascalVOC/dataset
EVAL_DIR=./logs/ssd_300_vgg_1_eval
CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1 \
    --max_num_batches=10

# =========================================================================== #
# Fine tune VGG-based SSD network
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/PascalVOC/dataset
TRAIN_DIR=/media/paul/DataExt4/PascalVOC/training/logs/ssd_300_vgg_6
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2012 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32

DATASET_DIR=/media/paul/DataExt4/PascalVOC/dataset
TRAIN_DIR=/media/paul/DataExt4/PascalVOC/training/logs/ssd_300_vgg_13
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2012 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32

DATASET_DIR=/media/paul/DataExt4/PascalVOC/dataset
TRAIN_DIR=/media/paul/DataExt4/PascalVOC/training/logs/ssd_300_vgg_2
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
CHECKPOINT_PATH=media/paul/DataExt4/PascalVOC/training/logs/ssd_300_vgg_1/
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2012 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.0005 \
    --learning_rate_decay_factor=0.96 \
    --batch_size=32

EVAL_DIR=${TRAIN_DIR}/eval
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=True \
    --batch_size=1


# =========================================================================== #
# Inception v3
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
DATASET_DIR=../datasets/ImageNet
TRAIN_DIR=./logs/inception_v3
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
CHECKPOINT_PATH=./checkpoints/inception_v3.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=4


CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/logs
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v3


# =========================================================================== #
# VGG 16 and 19
# =========================================================================== #
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/vgg_19.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --labels_offset=1 \
    --dataset_split_name=validation \
    --model_name=vgg_19


CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/vgg_16.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --labels_offset=1 \
    --dataset_split_name=validation \
    --model_name=vgg_16


# =========================================================================== #
# Xception
# =========================================================================== #
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=./logs/xception
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=xception \
    --labels_offset=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=600 \
    --save_interval_secs=600 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.0001 \
    --batch_size=32

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=xception \
    --labels_offset=1 \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=1


CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
CHECKPOINT_PATH=./logs/xception
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --labels_offset=1 \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=xception \
    --max_num_batches=10


CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.h5
python ckpt_keras_to_tensorflow.py \
    --model_name=xception_keras \
    --num_classes=1000 \
    --checkpoint_path=${CHECKPOINT_PATH}


# =========================================================================== #
# Dception
# =========================================================================== #
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=./logs/dception
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=dception \
    --labels_offset=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=32

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=dception \
    --labels_offset=1 \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=1


CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
CHECKPOINT_PATH=./logs/dception
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --labels_offset=1 \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=dception
