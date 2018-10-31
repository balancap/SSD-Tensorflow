#coding=utf-8
import os
import random

trainval_percent = 0.9
train_percent = 0.8
#路径修改为自己的路径
dir_pre="/media/hp/tyw/COCW_DATA/DetectionPatches_256x256/VOC2007/"
xmlfilepath = '/media/hp/tyw/COCW_DATA/DetectionPatches_256x256/VOC2007/Annotations/'
txtsavepath = '/media/hp/tyw/COCW_DATA/DetectionPatches_256x256/VOC2007/Main/'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(os.path.join(dir_pre,'ImageSets/Main/trainval.txt'), 'w')
ftest = open(os.path.join(dir_pre,'ImageSets/Main/test.txt'), 'w')
ftrain = open(os.path.join(dir_pre,'ImageSets/Main/train.txt'), 'w')
fval = open(os.path.join(dir_pre,'ImageSets/Main/val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
