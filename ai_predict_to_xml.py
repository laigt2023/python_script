from PIL import Image
from string import Template
import os
import sys
import requests
import json 
import base64
import draw_image
import report_event_image as REPORT_INSTANCE

# 用于进行AI模型推理/预标注，并输出xml文件

# AI图片推理端接口地址接口地址
AI_IP='192.168.19.240'
# AI图片识别任务ID  就模型ID
# AI_TASK_ID='e9e35528-71c7-4b62-9b88-8970bee55ab6'

# 新模型任务ID：8f97a571-22e5-41e3-afed-8f525d8338af
AI_TASK_ID='8f97a571-22e5-41e3-afed-8f525d8338af'

# 是否上报事件
IS_REPORT_EVENT = True

# AI图片推理端接口地址
def getAI_API_URL():
    return f'http://{AI_IP}:9090/api/inflet/v1/tasks/{AI_TASK_ID}/predict'

headers={"Content-Type": "application/json"}

# 文件最大数量
file_max_count = 1
# file文件计数器
file_count = 0
# 标注成功计数器
success_count = 0

# 是否打印消息
SHOW_LOG_MSG = True

# 禁用日志打印
def disableLogMsgOut():
    global SHOW_LOG_MSG
    SHOW_LOG_MSG = False
    
# 重置任务ID
def setAiTaskId(ai_task_id):
    global AI_TASK_ID
    AI_TASK_ID = ai_task_id

#  重设AI图片推理端接口地址
def setAiIp(ai_ip):
    global AI_IP
    AI_IP = ai_ip

# 清空输出目录
def emptyOutDir(out_dir): 
    if os.path.exists(out_dir):
        # os.remove(out_dir)
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                os.remove(os.path.join(root, file))
    else:
        os.makedirs(out_dir)

# AI图片推理端
# image_dir: 图片目录
# out_dir: 输出目录
# skip : 是否跳过已存在的xml文件
# IS_OUTPUT_JPEG  是否输出标注后的图片 目录为out_dir
# task_api_url AI服务地址 入参存在task_api_url 则优先使用task_api_url
# report_event_url 事件上报地址 默认:False-不上报 True-使用默认配置   填入url则使用URL
def ai_predict(image_dir,out_dir,skip,IS_OUTPUT_JPEG=False,task_api_url=False,report_event_url=False):
    try:
         # 重置计数器
        global file_count
        global file_max_count 
        global success_count 
    
        file_count = int(0)
        success_count = int(0)

        api_url = getAI_API_URL()

        # 入参存在task_api_url 则优先使用task_api_url
        if task_api_url:
            api_url = task_api_url

        for root, dirs, files in os.walk(image_dir):
            if files:
                # 记录文件列表总数
                file_max_count = files.__len__()

                logMsg(f'任务ID：{AI_IP}  目标模型检测推理任务  start...','false')
                for filename in files:
                    # 获取文件名称
                    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
                        
                        # 判断是否跳过已存在的xml文件
                        if skip == 'true':
                            xml_filename = imgae_name_to_xml_name(filename)
                            if os.path.exists(out_dir + os.path.sep + xml_filename):
                                logMsg(f'跳过已存在的xml文件：{xml_filename}')
                                continue


                        # 读取图片文件封装xml文件头部
                        image_path = os.path.join(image_dir, filename)
                        with Image.open(image_path) as img:
                            width, height = img.size
                            head_xml = head_xml_format(image_dir,filename,width,height)

                        # 读取图片文件封装xml标注内容
                        with open(image_path, "rb") as image_file:                           
                            encoded_string = base64.b64encode(image_file.read())
                            params = json.dumps({
                                'image':encoded_string.decode('utf-8')
                            })

                            # 发送请求
                            response = requests.post(api_url,params,headers)

                            # 检查响应状态码
                            if response.status_code == 200:
                                logMsg('请求成功','false')
                                result_data = response.json()
                                labels = result_data.get('data').get('targets')
                                
                                all_box_xml = ''
                                if labels == None:
                                    logMsg('暂无标签数据','true')
                                    continue
                                # 封装xml文件画框内容坐标
                                for index,label in enumerate(labels):
                                    box = label.get('bbox')
                                    if box:
                                        label_name = box.get('label')
                                        label_box= box.get('box')
                            
                                        bbox_data = { 
                                            'name' : label_name, 
                                            'x_min' : int(float(label_box.get("left_top_x"))), 
                                            'y_min' : int(float(label_box.get("left_top_y"))), 
                                            'x_max' : int(float(label_box.get("right_bottom_x"))), 
                                            'y_max' : int(float(label_box.get("right_bottom_y")))
                                        }
                                        xml=bbox_to_temple(bbox_data)
                                        all_box_xml = all_box_xml + xml
                                    else:
                                        logMsg('暂无标签数据','false')

                                complete_xml = head_xml + all_box_xml + head_xml_end_format() 
                                # 生成xml文件
                                create_xml_file(out_dir,filename,complete_xml)

                                # 生成带有标签的jpeg图片
                                if IS_OUTPUT_JPEG == True:
                                    inputImage = image_dir + os.path.sep + filename
                                    outputImage = out_dir + os.path.sep + filename.replace('.jpg','.jpeg')
                                    message,out_url = draw_image.draw_image(inputImage,outputImage,labels)
                                    logMsg(f'{ message }','false')

                            else:
                                logMsg(f'请求失败，文件{ filename }','false')
                                logMsg(f'请求失败，状态码：{response.status_code} - {response.text}','false')
                                continue
                logMsg(f'任务ID：{AI_IP}  目标模型检测推理任务  end...','false')  

                if IS_OUTPUT_JPEG and report_event_url:
                    # 上报事件
                    # 使用脚本的额默认report_url
                    REPORT_INSTANCE.report_event_image(out_dir,True)
                           

    except requests.exceptions.RequestException as e:
        logMsg(f'请求发生异常：{str(e)}')
        logMsg(f'任务ID：{AI_IP}  目标模型检测推理任务  end...','false')     

# 打印消息
def logMsg(text,isOperate='true'):
    global file_count
    global file_max_count
    global success_count
    global SHOW_LOG_MSG

    if isOperate != 'false':
        file_count = int(file_count) + 1

    if SHOW_LOG_MSG:    
        print(f'已标注成功{success_count}/{file_max_count}(标注数/总数)  当前进度:{file_count}/{file_max_count}: {text}')

# 生成xml文件
def create_xml_file(xml_file_path,image_name,xml_content):
    global success_count 
    xml_filename = imgae_name_to_xml_name(image_name)
    
    # 创建目录
    if not os.path.exists(xml_file_path):
        os.makedirs(xml_file_path)

    # os.path.sep: 获取当前系统的路径分隔符
    xml_file_path = xml_file_path + os.path.sep + xml_filename
    # 写入文件
    with open(xml_file_path, 'w', encoding='utf-8') as xml_file:
        xml_file.write(xml_content)
        success_count = int(success_count) + 1
        logMsg(xml_file_path + ' 文件生成成功')    

# 图片格式名称转XML格式名称
def imgae_name_to_xml_name(image_name):
    if image_name.lower().endswith('.jpg'):
        image_name = image_name.replace('.jpg','.xml')
        image_name = image_name.replace('.JPG','.xml')
    elif image_name.lower().endswith('.jpeg'):
        image_name = image_name.replace('.jpeg','.xml')
        image_name = image_name.replace('.JPEG','.xml')

    elif image_name.lower().endswith('.png'):
        image_name = image_name.replace('.png','.xml')
        image_name = image_name.replace('.PNG','.xml')

    return image_name

# 封装xml文件结尾
def head_xml_end_format():
    xml_end = '''
    </annotation>
    '''
    return xml_end

# 封装xml文件头部
def head_xml_format(image_dir,filename,width,height,):
     # 定义json模板
    #建立xml文件框架
    xml_head = '''<annotation>
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
    img_data = { 'path' : image_dir + filename, 'filename' : filename, 'width' : int(float(width)), 'height' : int(float(height)) }
    check_img_data(img_data)
    head_xml_text = Template(xml_head).substitute(img_data)
    return head_xml_text

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

# 检测数据项是否为空
# obj: 数据对象
def isNotEmpty(obj, key):
    if key in obj:
        if obj[key] == '':
            raise Exception(file_name + ':  ' + key +' is null')
        return True    
    else:
        raise Exception(file_name + ':  ' + key +' is null')

    # 坐标点封装模板
def bbox_to_temple(obj):
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

# 主函数
if __name__ == "__main__":
    # 输入校验
    if(sys.argv.__len__() < 2):
        logMsg('请输入参数1: 图片目录')
        exit() 

    # 推理图片目录
    img_dir = sys.argv[1]
    # 输出xml目录
    out_dir = './out_xml'
    # 是否跳过已存在的xml文件     true-跳过，false-不跳过，默认true
    skip = 'true'
    is_output_jpeg = False
    report_event_url = False
    task_api_url = ''
    if sys.argv.__len__() > 2 and sys.argv[2] != 'null':
        out_dir = sys.argv[2]

    if sys.argv.__len__() > 3 and sys.argv[3]!= 'null':    
        skip = sys.argv[3] 

    if sys.argv.__len__() > 4 and sys.argv[4]!= 'null':
        if sys.argv[4] == 'true':    
            is_output_jpeg = True 
        else:
            is_output_jpeg = False

    if sys.argv.__len__() > 5 and sys.argv[5]!= 'null':
        task_api_url = sys.argv[5]      

    if sys.argv.__len__() > 6 and sys.argv[6]!= 'null':
        report_event_url = sys.argv[6] 

    print(sys.argv)
    ai_predict(img_dir,out_dir,skip,is_output_jpeg,task_api_url,report_event_url)