import json, os
from PIL import Image
from xml.dom.minidom import parse
from tqdm import tqdm


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
        with open(t_p, "r") as t:
            bboxs = t.readlines()
            for bbox in bboxs:
                bbox = bbox.strip().split(' ')
                label = eval(bbox[0].strip()) #标签编号

                # if label != 2:  # 保留需要的标签id
                #     continue
                # xmin, ymin, xmax, ymax = list(map(eval, bbox[1:]))

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

        with open(x_p, "w", encoding='utf-8') as xml_f:
            xml_f.write(head + obj + xml_end)
        cnt += 1
        print(f"convert success {cnt} xml")

# voc标准的xml
def xml2json(xml_path, jsonfile, img_path='/opt/gongdi/biaozhu/image/val'):
    cat2id, img2id = {}, {}
    categories = []
    images = []
    annotations = []
    cat_count, img_count, ann_count = 0, 0, 0

    xmls = os.listdir(xml_path)
    for xml in tqdm(xmls):
        # print(xml)
        root = parse(os.path.join(xml_path, xml))
        filename = root.getElementsByTagName('filename')[0].firstChild.data #获取xml中的文件名

        # 为防止原xml中的filename有改动，改成与xml对应的图片文件名
        name, ext = os.path.splitext(filename)
        # print(name, ext)
        filename = xml.replace('.xml', ext)

        '''
        编辑图片
        '''
        if img_path is not None:
            f_p = os.path.join(img_path, filename)

            if not os.path.exists(f_p):
                continue
            
            img = Image.open(f_p)
            W, H = img.size

        if filename not in img2id:
            img2id[filename] = img_count
            # size = root.getElementsByTagName('size')[0]
            # width = float(size.getElementsByTagName('width')[0].firstChild.data)
            # height = float(size.getElementsByTagName('height')[0].firstChild.data)
            images.append(dict(file_name=filename, id=img2id[filename]))
            img_count += 1  #迭代img_id

        objects = root.getElementsByTagName('object')  #检测框信息

        for obj in objects:  #一张图有多个object（检测框）

            try:
                cat = obj.getElementsByTagName('name')[0].firstChild.data
                label = ['person']
                # if len(cat)==0:
                #     continue
                # if cat not in label:
                #     continue
                # else:
                #     continue
                # elif cat == '瓶身破损':
                #     cat ='ps_breakage'
                # elif cat == '瓶盖变形':
                #     cat ='deformation'
                # elif cat == '瓶盖坏边':
                #     cat = 'bad_selvedge'
                # elif cat == '瓶盖打旋':
                #     cat ='swirling'
                # elif cat == '瓶盖断点':
                #     cat = 'breakpoint'
                # elif cat == '标贴歪斜':
                #     cat ='skew'
                # elif cat == '标贴起皱':
                #     cat = 'wrinkle'
                # elif cat == '标贴气泡':
                #     cat ='bt_bubble'
                # elif cat == '喷码正常':
                #     cat = 'normal'
                # elif cat == '喷码异常':
                #     cat ='abnormal'
                # elif cat == '瓶身气泡':
                #     cat = 'ps_bubble'
                # elif cat == '酒液杂质':
                #     cat ='impurity'
                # else:
                #         continue

                '''
                此处可对类别进行预处理
                '''

            # bbox = obj.getElementsByTagName('bndbox')[0]
                xmin = int(float(obj.getElementsByTagName('xmin')[0].firstChild.data))
                ymin = int(float(obj.getElementsByTagName('ymin')[0].firstChild.data))
                xmax = int(float(obj.getElementsByTagName('xmax')[0].firstChild.data))
                ymax = int(float(obj.getElementsByTagName('ymax')[0].firstChild.data))
                '''
                此处可对框标注信息进行处理
                '''
                # w, h = xmax-xmin, ymax-ymin
                # area=w*h
                # if area / (W*H) < 0.00028:
                #     continue
            except Exception as e:
                continue


            if cat not in cat2id: # category_id
                cat2id[cat] = cat_count
                categories.append(dict(id=cat_count, name=cat))
                cat_count += 1
            ann = dict(id=ann_count, image_id=img2id[filename], category_id=cat2id[cat],
                     bbox=[xmin, ymin, xmax-xmin, ymax-ymin],
                    area=(xmax - xmin) * (ymax - ymin))
                    #  bbox=[xmin, ymin, xmax, ymax],
                    # area=xmax * ymax)
            annotations.append(ann)
            ann_count += 1
    names = os.listdir(img_path)
    n = len(img2id)
    images=[]
    print(img2id)
    for filename in names:

        _id = img2id.get(filename)
        if _id==0:
            pass
        elif not _id:
            _id = n
            n = n+1
            # print(img2id.get(filename))
            print(_id)
        info = dict(
            file_name = filename,
            id = _id
        )
        images.append(info)

    with open(jsonfile, 'w', encoding='utf-8') as f:
        json_infos = dict(categories=categories, images=images, annotations=annotations)
        json.dump(json_infos, f, indent=2, ensure_ascii=False)

#COCO标准的json
def json2xml(jsonfile, xml_path):
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)

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
    img_res = {}
    with open(jsonfile, "r", encoding='utf-8') as fs:
        info = json.load(fs)
        cat_id = [x['id'] for x in info['categories']]
        cat_n = [x['name'] for x in info['categories']]
        catIdname = dict(zip(cat_id, cat_n))

        img_id = [x['id'] for x in info['images']]
        img_n = [x['file_name'] for x in info['images']]
        imgIdname = dict(zip(img_id, img_n))

        for i, ann in enumerate(info['annotations']):
            image_id = ann['image_id']
            file_name = imgIdname[image_id]
            y_id = ann['category_id']
            cat_name = catIdname[y_id]
            ann['category_name'] = cat_name  #添加类名的键
            if file_name not in img_res:
                img_res[file_name] = [ann]
            else:
                img_res[file_name].append(ann)
    cnt = 0
    for img in img_res.keys():
        _, ext = os.path.splitext(img)
        head = xml_head.format(img)
        anns = img_res[img]
        obj = ''
        x_p = os.path.join(xml_path, img.replace(ext, '.xml'))
        for ann in anns:
            label = ann['category_name']
            xmin, ymin, w, h = ann['bbox']
            xmax, ymax = xmin+w, ymin+h
            obj += xml_obj.format(label, xmin, ymin, xmax, ymax)

        with open(x_p, "w", encoding='utf-8') as x:
            x.write(head + obj + xml_end)
        cnt += 1
        print(f"convert success {cnt} xml")

def xml2txt(xml_path, txt_path, img_path, label_l):  #label_l为列表，index对应label编号
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    xmls = os.listdir(xml_path)
    for xml in tqdm(xmls):
        xml_p = os.path.join(xml_path, xml)
        txt_p = os.path.join(txt_path, xml.replace('xml','txt'))
        root = parse(xml_p)
        filename = root.getElementsByTagName('filename')[0].firstChild.data  # 获取xml中的文件名
        img_p = os.path.join(img_path, filename)
        img = Image.open(img_p)
        W, H = img.size
        objects = root.getElementsByTagName('object')  # 检测框信息
        for obj in objects:  #一张图有多个object（检测框）
            try:
                cat = obj.getElementsByTagName('name')[0].firstChild.data
                if cat not in label_l:
                    continue
                    # id = 2
                # else:
                id = label_l.index(cat)
                # if cat != 'other_fire':
                #     cat = 'other_fire'
            except Exception as e:
                print(e)
                continue
            # bbox = obj.getElementsByTagName('bndbox')[0]
            xmin = int(float(obj.getElementsByTagName('xmin')[0].firstChild.data))
            ymin = int(float(obj.getElementsByTagName('ymin')[0].firstChild.data))
            xmax = int(float(obj.getElementsByTagName('xmax')[0].firstChild.data))
            ymax = int(float(obj.getElementsByTagName('ymax')[0].firstChild.data))
            # 转为yolo 格式
            x = (xmin+xmax)/(W*2.0)
            y = (ymin+ymax)/(H*2.0)
            w = (xmax-xmin)/W
            h = (ymax-ymin)/H
            with open(txt_p, 'a', encoding='utf-8') as t:
                t.write(f"{id} {x} {y} {w} {h}\n")





if __name__ == "__main__":
    #json_p = r"F:\DATA\QST-data\Error_detect_1\原始数据\smoke\annotations"
    #txt_path = r"E:\2月任务\2.3\共达地-人和手机标注-23.02.02\共达地-人和手机标注-23.02.02\phone\txt"
    #xml_path = r"D:\movies\TD_BLUE_HELMET\trainxml"
    xml_path = '/opt/gongdi/biaozhu/1080P/xml/val'
    labels = ['TD_blue_helmet']
    # classes = ['phone', 'smoke', 'smoking', 'sit', 'down', 'fire']
    #img_path = r"D:\movies\TD_BLUE_HELMET\train"
    img_path = '/opt/gongdi/biaozhu/image/val'
    #jsonfile = r"D:\movies\TD_BLUE_HELMET\train.json"
    jsonfile = '/opt/gongdi/biaozhu/image/val.json'
    # json2txt(json_p, txt_path)
    # txt2xml(txt_path, xml_path, labels, img_path)
    xml2json(xml_path, jsonfile)
    # json2xml(jsonfile, xml_path)

    # xml2txt(xml_path, txt_path, img_path, labels)




