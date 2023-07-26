from PIL import Image
from string import Template
import os
import sys

# 标签名称
dimension_label = 'bicycle'


# 坐标点封装模板
def text_to_temple(obj):
    check_obj_data(obj)

    xml_obj = '''
        <object>
            <name>$name</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>$x_min</xmin>
                <ymin>$y_min</ymin>
                <xmax>$x_max</xmax>
                <ymax>$y_max</ymax>
            </bndbox>
        </object>
    '''
    return Template(xml_obj).substitute(obj)

# image_dir: 图片目录
# label_directory: label文件目录
# file_name: 文件名称
def img_file_to_xml(image_dir ,label_directory, file_name ,out_directory):
    image_dir = join_file_separator(image_dir)
    label_directory = join_file_separator(label_directory)
    out_directory = join_file_separator(out_directory)
    
    image_file_name = ''
    # 支持的图片格式
    image_formats = ['.jpg', '.jpeg','.png']
    # 读取图片文件
    for img_suffix in image_formats:
        img_path = image_dir + file_name + img_suffix
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                # 获取图片的宽度和高度
                width, height = img.size
                image_file_name = file_name + img_suffix
                  

    # 读取label文件                
    with open(label_directory + file_name +'.txt', 'r') as text_file:
        text = text_file.read()
        object_xml_text = ''
        if text:
            lines = text.split('\n')
            for index,line in enumerate(lines):
                data = line.split(' ')
      
                if line:
                    obj_data = { 
                        'name' : data[0].lower(), 
                        'x_min' : int(float(data[1])), 
                        'y_min' : int(float(data[2])), 
                        'x_max' : int(float(data[3])), 
                        'y_max' : int(float(data[4]))
                    }
                    object_xml_text += text_to_temple(obj_data)

    # 定义json模板
    #建立xml文件框架
    xml_head = '''<annotation>
        <folder>Bicycle</folder>
        <filename>$filename</filename>
        <path>$path</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>$width</width>
            <height>$height</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        '''
    xml_end = '''
    </annotation>
    '''
    img_data = { 'path' : image_dir + image_file_name, 'filename' : image_file_name, 'width' : int(float(width)), 'height' : int(float(height)) }
    check_img_data(img_data)
    head_xml_text = Template(xml_head).substitute(img_data)

    content=head_xml_text + object_xml_text + xml_end

    # 创建xml文件
    create_xml_file(out_directory + file_name + '.xml',content)

# 创建xml文件并写入内容
def create_xml_file(out_file_path , xml_content):
    # 创建目录
    directory = os.path.dirname(out_file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # 写入文件
    with open(out_file_path, 'w', encoding='utf-8') as xml_file:
        xml_file.write(xml_content)
        print(out_file_path + ' 文件生成成功')    

# 检测数据项是否为空
# obj: 数据对象
def isNotEmpty(obj, key):
    if key in obj:
        if obj[key] == '':
            raise Exception(file_name + ':  ' + key +' is null')
        return True    
    else:
        raise Exception(file_name + ':  ' + key +' is null')

# 检测图片数据是否完整
# img_data: 图片数据
def check_img_data(img_data):
    result = True
    if isNotEmpty(img_data,'filename'):
        result = False
    if isNotEmpty(img_data,'path'):
        result = False      
    if isNotEmpty(img_data,'width'):
        result = False
    if isNotEmpty(img_data,'height'):
        result = False         
    return result

# 检测标签数据是否完整
# obj_data: 标签数据
def check_obj_data(obj_data):
    result = True
    if isNotEmpty(obj_data, 'name'):
        result = False
    if isNotEmpty(obj_data,'x_min'):
        result = False
    if isNotEmpty(obj_data,'y_min'):
        result = False
    if isNotEmpty(obj_data,'x_max'):
        result = False
    if isNotEmpty(obj_data,'y_max'):
        result = False       
    return result


# 按照平台拼接文件分隔符
def join_file_separator(path):
    if path.endswith(os.path.sep):
        return path
    return path + os.path.sep

# mian
# 例子命令 python ./label_to_json.py D:\工作交接\智慧工地\训练集\open-image-dataset\tool\OIDv4_ToolKit\OID\Dataset\train\Bicycle D:\工作交接\智慧工地\训练集\open-image-dataset\tool\OIDv4_ToolKit\OID\Dataset\train\Bicycle\Label D:\工作交接\智慧工地\训练集\open-image-dataset\tool\OIDv4_ToolKit\OID\Dataset\train\Bicycle\xml

if __name__ == '__main__':
    # 检测参数
    if(sys.argv.__len__() < 4):
        print(sys.argv.__len__())
        if sys.argv.__len__() < 2:
            print('请输入参数1: 图片目录')
         
        if sys.argv.__len__() < 3:
            print('请输入参数2:label文件目录')
           
        if sys.argv.__len__() < 4:
            print('请输入参数3:输出目录')
        exit() 
  
    # 图片目录  如：'D:\\工作交接\\智慧工地\训练集\\open-image-dataset\\tool\\OIDv4_ToolKit\\OID\Dataset\\train\\Bicycle'
    image_dir = sys.argv[1]
    # label文件目录 如：'D:\\工作交接\\智慧工地\训练集\\open-image-dataset\\tool\\OIDv4_ToolKit\\OID\Dataset\\train\\Bicycle\\Label'
    label_dir =  sys.argv[2] 
    # 输出目录 如：'D:\\工作交接\\智慧工地\训练集\\open-image-dataset\\xml\\label'
    out_directory =  sys.argv[3] 
    
    # label文件计数器
    file_count = 0
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            # 获取文件名称
            if file.endswith('.txt'):
                file_name = file.split('.')[0]
                file_count += 1
              
                # 输出全部
                img_file_to_xml(image_dir, label_dir,file_name,out_directory)       

    print('总：' + str(file_count) + '个文件')
 