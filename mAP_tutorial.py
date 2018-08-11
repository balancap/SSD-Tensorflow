#coding=utf-8
"""
参照voc_detection的标准的mAP计算方法
用来计算我们自己的数据的值，
mean Average Precision指的是多个类的AP的平均，
怎么平均的mAP,
怎么弄呢，其实就是将多个类的AP求和，然后进行平均，即可得到检测的mAP值。
"""
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from pylab import mpl
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

#如果是self_data数据，则需要对label进行转化，讲中心点加宽高转化为左上加右下
def read_file(file,classname):
    with open(file,'r') as f:
        lines=f.readlines()
    if classname==-1:
        lines=[line.strip().split(" ") for line in lines]
        bboxes_to=[]
        for item in lines:
            bboxes_to.append(item[1:])
    else:
        lines_=[line.strip().split(" ") for line in lines]
        lines=[]
        bboxes_to=[]
        for i,line in enumerate(lines_):
            if int(line[0])==classname:
                lines.append(line)
                bboxes_to.append(line[1:])
                # print(bboxes_to_[i])
    # bboxes=[]
    # for bbox in bboxes_to:
    #     item=[float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])]
    #     bboxes.append(item)
    # return np.array(bboxes),lines
    bboxes = []
    for bbox in bboxes_to:
        item = [float(bbox[0])-float(bbox[2])/2.0, float(bbox[1])-float(bbox[3])/2.0, float(bbox[0])+float(bbox[2])/2.0, float(bbox[1])+float(bbox[3])/2.0]
        bboxes.append(item)
    return np.array(bboxes),lines

def convert(gt_dir,classname):
    class_recs={}
    npos=0
    for root,dirs,files in os.walk(gt_dir):
        for i,file in enumerate(files):
            cur_path=os.path.join(root,file)
            bbox,R=read_file(cur_path,classname)
            if classname!=-1:
                det=[False]*len(R)
                npos+=len(R)
                class_recs[file]={"bbox":bbox,'det':det}
            else:
                gt_cls_id=[]
                for item in R:
                    gt_cls_id.append(item[0])
                det = [False] * len(R)
                npos += len(R)
                class_recs[file] = {"bbox": bbox, 'det': det, "gt_cls_id":gt_cls_id}
            print("正在转化中。。。"+str(len(files)-i))
    return class_recs,npos
#更加详细的资料可以查看https://github.com/Tangzixia/Object-Detection-Metrics#average-precision
#计算某个类的AP,class=-1时候代表计算所有类的mAP
def gen_ap(gt_dir,pred_res,classname,iou=0.5):
    class_recs,npos=convert(gt_dir,classname)
    with open(pred_res,'r') as f:
        lines=f.readlines()
    #img_id,confidence,BB,
    #得分
    splitlines=[item.strip().split(" ") for item in lines]
    img_ids=[x[0] for x in splitlines]
    cls_flgs=np.array([x[1] for x in splitlines])
    confidence=np.array([float(x[2]) for x in splitlines])
    BB=np.array([[float(z) for z in x[3:]] for x in splitlines])

    # 找出每一类对应的预测候选框
    # 如果是classname==-1,则说明需要计算所有类的mAP值，这时候需要得到所有的类别标签
    if classname!=-1:
        inds=np.zeros(len(splitlines))
        for i,item in enumerate(splitlines):
            if int(item[1])==classname:
                inds[i]=1
        img_ids_=[]
        confidence_=[]
        BB_=[]
        for i,item in enumerate(splitlines):
            if inds[i]==1:
                img_ids_.append(img_ids[i])
                confidence_.append(confidence[i])
                BB_.append(BB[i])
        img_ids=img_ids_
        confidence=np.array(confidence_)
        BB=np.array(BB_)
        # img_ids=list(np.array(img_ids[np.array(inds)]))
        # confidence=list(np.array(confidence[np.array(inds)]))
        # BB=list(np.array(BB[np.array(inds)]))


        #confidence由大到小排序
        sorted_ind=np.argsort(-confidence)
        # np.argsort(-confidence<=-.3)
        sorted_ind1 = np.where(confidence[sorted_ind] >= .0)[0]
        sorted_ind = sorted_ind[sorted_ind1]
        print(len(sorted_ind))
        BB=BB[sorted_ind,:]
        img_ids=[img_ids[x] for x in sorted_ind]

        # sorted_ind = np.argsort(-confidence)
        # print(len(sorted_ind))
        # BB = BB[sorted_ind, :]
        # img_ids = [img_ids[x] for x in sorted_ind]

        nd=len(img_ids)
        print(nd)
        tp=np.zeros(nd)
        fp=np.zeros(nd)

        for d in range(nd):
            R=class_recs[img_ids[d]]
            bb=BB[d,:].astype(float)
            ovmax = -np.inf
            BBGT=R['bbox'].astype(float)

            if BBGT.size>0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                overlaps=inters/uni
                ovmax=np.max(overlaps)
                # print(ovmax)
                jmax=np.argmax(overlaps)
            if ovmax>iou:
                if not R['det'][jmax]:
                    tp[d]=1
                    R['det'][jmax]=1
                else:
                    fp[d]=1
            else:
                fp[d]=1
    else:
        # confidence由大到小排序
        sorted_ind = np.argsort(-confidence)
        # np.argsort(-confidence<=-.3)
        sorted_ind1 = np.where(confidence[sorted_ind] >= .3)[0]
        sorted_ind = sorted_ind[sorted_ind1]
        BB = BB[sorted_ind, :]
        img_ids = [img_ids[x] for x in sorted_ind]
        cls_flgs=cls_flgs[sorted_ind]

        # sorted_ind = np.argsort(-confidence)
        # print(len(sorted_ind))
        # BB = BB[sorted_ind, :]
        # img_ids = [img_ids[x] for x in sorted_ind]

        nd = len(img_ids)
        print(nd)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            R = class_recs[img_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                # print(ovmax)
                jmax = np.argmax(overlaps)
            if ovmax > iou and R['gt_cls_id'][jmax]==cls_flgs[d]:
                if not R['det'][jmax]:
                    tp[d] = 1
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1
            else:
                fp[d] = 1

    fp=np.cumsum(fp)
    tp=np.cumsum(tp)
    rec=tp/float(npos)
    prec=tp/np.maximum(tp+fp,np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec)
    return rec,prec,ap
def draw_plot(rec,prec,ap,name,path):
    if os.path.exists(path)==False:
        os.mkdir(path)
    myfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK.ttc")
    mpl.rcParams['axes.unicode_minus'] = False
    tick=np.arange(0,1.1,0.1)
    plt.figure()
    plt.title(name+":"+str(ap),fontproperties=myfont)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0,1,0,1.05])
    plt.xticks(tick)
    plt.yticks(tick)
    plt.plot(rec,prec)
    # plt.show()

    plt.savefig(os.path.join(path,name+".png"))
if __name__=="__main__":
    gt_dir = "/home/hp/Data/house_data/train/Data_valid/labels_initial/"
    pred_res = "../res_self_data_0.0.txt"
    mAP_file="/home/hp/Desktop/yolov3-res/map.txt"

    dict_ = {"0": u"迷彩建筑", "1": u"一般建筑", "2": u"迷彩油罐", "3": u"一般油罐", "4": u"迷彩雷达", "5": u"一般雷达"}
    ap_list=[]
    for i in range(6):
        classname =i
        rec, prec, ap = gen_ap(gt_dir, pred_res, classname)
        draw_plot(rec,prec,ap,dict_[str(classname)],path="/home/hp/Desktop/yolov3-res/")
        ap_list.append(ap)
        print(rec, prec, ap)
    with open(mAP_file,'w') as f:
        for i,ap in enumerate(ap_list):
            f.write(str(dict_[str(i)].decode('utf8'))+":"+str(ap)+"\n")
        f.write("mAP:"+str(round(np.array(ap_list).mean(),4)))
    print("mAP50的值为：",round(np.array(ap_list).mean(),4))
