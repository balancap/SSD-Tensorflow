#coding=utf-8
import glob
import os

s1="""    <object>
        <name>{0}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{1}</xmin>
            <ymin>{2}</ymin>
            <xmax>{3}</xmax>
            <ymax>{4}</ymax>
        </bndbox>
    </object>"""

s2="""<annotation>
    <folder>VOC2007</folder>
    <filename>{0}</filename>
    <source>
        <database>My Database</database>
        <annotation>VOC2007</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>J</name>
    </owner>
    <size>
        <width>512</width>
        <height>512</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>{1}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{2}</xmin>
            <ymin>{3}</ymin>
            <xmax>{4}</xmax>
            <ymax>{5}</ymax>
        </bndbox>
    </object>{6}
</annotation>
"""
# dict_={0:"micai_jianzhu",1:"yiban_jianzhu",2:"micai_youguan",3:"yiban_youguan",4:"micai_leida",5:"yiban_leida"}
# dict_={1:"car_1",2:"car_2",3:"car_3",4:"car_4"}
dict_={0:"camouflage_car",1:"non_camouflage_car"}
def convert2xml(label_dir,dst_dir_xml):
	if os.path.exists(dst_dir_xml)==False:
		os.mkdir(dst_dir_xml)
	textlist=glob.glob(os.path.join(label_dir,'*.txt'))
	# print(len(textlist))
	for text_ in textlist:
		flabel = open(text_, 'r')
		lb = flabel.readlines()
		flabel.close()
		lb=[line.strip() for line in lb]
		ob2 = ""
		x1=lb[0].split(' ')[0]
		x1=dict_[int(x1)]
    #注意这里如果给定的是中心点的坐标和宽高，需要转化为左上角和右下角的坐标，否则则不转化
		x3=lb[0].split(" ")[1:]
		x3=[int(float(x3[0])-float(x3[2])/2.0),int(float(x3[1])-float(x3[3])/2.0),int(float(x3[0])+float(x3[2])/2.0),int(float(x3[1])+float(x3[3])/2.0)]

		if len(lb)>1:  # extra annotation
			for i in range(1,len(lb)):
				cls_id=lb[i].split(' ')[0]
				cls_id=dict_[int(cls_id)]
				x3_tmp=lb[i].split(' ')[1:]
				x3_tmp=[int(float(x3_tmp[0])-float(x3_tmp[2])/2.0),int(float(x3_tmp[1])-float(x3_tmp[3])/2.0),int(float(x3_tmp[0])+float(x3_tmp[2])/2.0),int(float(x3_tmp[1])+float(x3_tmp[3])/2.0)]
				ob2+='\n' + s1.format(cls_id,x3_tmp[0],x3_tmp[1],x3_tmp[2],x3_tmp[3])
		# imgname=text_.split("/")[-1].split(".")[0]+'.jpg'
		# savename=os.path.join(dst_dir_xml,text_.split("/")[-1].split(".")[0]+'.xml')
		tmp_name=text_.split("/")[-1].split(".")
		pre_name=""
		for i in range(len(tmp_name)-1):
			pre_name=pre_name+tmp_name[i]+"."
		# print(pre_name)
		imgname = pre_name+'jpg'
		savename = os.path.join(dst_dir_xml,pre_name+'xml')
		f = open(savename, 'w')
		ob1=s2.format(imgname, x1, x3[0],x3[1],x3[2],x3[3], ob2)
		f.write(ob1)
		f.close()

if __name__=="__main__":
	# label_dir="/media/hp/tyw/COCW_DATA/DetectionPatches_256x256/VOC2007/labels_std"
	# dst_dir_xml="/media/hp/tyw/COCW_DATA/DetectionPatches_256x256/VOC2007/Annotations"
	label_dir="/home/hp/Data/car_train_data/labels_initial_csv"
	dst_dir_xml="/home/hp/Data/car_train_data/VOC2007/Annotations"
	convert2xml(label_dir,dst_dir_xml)
