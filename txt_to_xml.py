from PIL import Image
from string import Template
import os
import sys

# yolo格式的txt
def txt2xml(txt_path, xml_path, labels, img_path, img_ext='.jpg'):
    """
    labels参数为列表，索引与标签id相同
    """
    if not os.path.exists(xml_path):
        os.mkdir(xml_path)
    #建立xml文件框架
    xml_head = '''<annotation>
    	<folder>train</folder>
    	<filename>{}</filename>
    	<source>
    		<database>Unknown</database>
    	</source>
    	<segmented>0</segmented>
        '''
    xml_obj = '''
    <object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
    '''
    xml_end = '''
    </annotation>
    '''
    cnt = 0
    txts = os.listdir(txt_path)
    for txt in txts:
        name, ext = os.path.splitext(txt)
        filename = name + img_ext
        head = xml_head.format(filename)  #填入xml中的filename
        img = Image.open(os.path.join(img_path, filename))
        img_w, img_h = img.size  #获取图片宽高，用于坐标转化
        t_p = os.path.join(txt_path, txt)
        x_p = os.path.join(xml_path, txt.replace('txt', 'xml')) #xml文件
        obj = ''
        is_not_empty = False
        with open(t_p, "r") as t:
            bboxs = t.readlines()
            for bbox in bboxs:
                bbox = bbox.strip().split(' ')
                label = eval(bbox[0].strip()) #标签编号

                if labels[label] == 'safetyVest':  # 保留需要的标签id  txt中出现多个标记记录，这里进行赛选需要的【safetyVest】-安全衣/反光衣标签
                    is_not_empty = True          
                    x_center = round(float(str(bbox[1]).strip()) * img_w) #round 去掉小数部分
                    y_center = round(float(str(bbox[2]).strip()) * img_h)
                    bbox_w = round(float(str(bbox[3]).strip()) * img_w)
                    bbox_h = round(float(str(bbox[4]).strip()) * img_h)
                    #计算bbox的左上右下坐标
                    xmin = str(int(x_center - bbox_w / 2)) #转为str填入xml_obj
                    ymin = str(int(y_center - bbox_h / 2))
                    xmax = str(int(x_center + bbox_w / 2))
                    ymax = str(int(y_center + bbox_h / 2))

                    obj += xml_obj.format(labels[label], xmin, ymin, xmax, ymax)

        if is_not_empty == True:
            with open(x_p, "w", encoding='utf-8') as xml_f:
                xml_f.write(head + obj + xml_end)
                cnt += 1
                print(f"convert success {cnt} xml")

if __name__ == "__main__":
    txt_path = r"D:\工作交接\智慧工地\训练集\SafetyVest\txt"  #yolo格式的txt
    xml_path = r"D:\工作交接\智慧工地\训练集\SafetyVest\xml"  #转换后的xml路径
    labels = ['helmet','safetyVest']                        #需要的标签
    img_path = r'D:\工作交接\智慧工地\训练集\SafetyVest\img'  #图片路径
    print(txt_path)
    print(xml_path)
    print(img_path)
    txt2xml(txt_path, xml_path, labels, img_path)