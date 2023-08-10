import requests
import sys
import os
import time
import base64
import json
import xml.etree.ElementTree as ET

# 上报事件类型汇总，helmet-安全帽，vest-反光衣  事件类型：0-安全帽监测、1-反光衣监测、15-安全帽+人脸识别、16-反光衣+人脸识别
AI_TYPES=[{"key":'helmet',"eventType":"0"},{"key":'vest',"eventType":"1"}]

# 文件夹后缀
SUFFIX='_xml'

def report_event_image(image_dir, report_url):
    global AI_TYPES
    global SUFFIX

    array=image_dir.split(os.path.sep)
    dir_name = array[array.__len__() - 1]

    current_ai_type_key = ''
    current_ai_type_event_id = ''
    fileName=''
    cameraName=''

    # 文件名称
    
    for t in AI_TYPES:
        if image_dir.endswith("_" + t["key"] + SUFFIX):
            current_ai_type= t["key"]
            current_ai_type_event_id= t["eventType"]
            fileName= dir_name.replace("_" + t["key"] + SUFFIX,'')
            break

    if current_ai_type == '':
        print('未找到对应的AI类型:' + image_dir)
        return

    _array=fileName.split("_")
    if _array.__len__() > 1:
        cameraName=_array[0] + "_" + _array[1]
    elif _array.__len__() == 1:
        cameraName=_array[0]
    

    post_data = {
        # 工地ID
        "siteID" : array[array.__len__() - 3],
        # 日期
        "alarmDate" : array[array.__len__() - 2],
        # 告警视频名称
        "videoName" : fileName + '.mp4',
        # 告警类型 'helmet' - 安全帽 'vest' - 反光衣
        "alarmType" : current_ai_type_event_id,

        # 安全帽名称
        "cameraName":cameraName,
        # 
        # 发件告警的时间戳 "alarmTime" : int(round(time.time() * 1000)),
        # 人脸识别信息：身份证号
        # 告警详情（JSONArray对象数组）  [{ "cardId":"", "alarmTime":"" , "alarmPicture":"" }]
        "info":[]
    }

    # 发件告警的时间戳
    alarmTime = int(round(time.time() * 1000))

    # 读取目标模型的jepg文件
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.endswith('.jpeg'):
                # 读取图片文件
                jpeg_file_path = root + os.path.sep + f
                post_data["info"] = []
                with open(jpeg_file_path, 'rb') as jpeg_file:
                    if jpeg_file:
                        # 构建请求参数
                        image_base_code = base64.b64encode(jpeg_file.read()).decode('utf-8')

                        xml_file_path = jpeg_file_path.replace('.jpeg', '.xml')
                        target_count = 1

                        # 从xml文件中获取目标数量
                        with open(xml_file_path, 'rb') as xml_file:
                            if xml_file:
                                tree = ET.parse(xml_file_path)
                                xml_root = tree.getroot()

                                # 获取特定元素的数量
                                target_count = len(xml_root.findall('object'))
                                print(f"共识别识别了{target_count}个目标({xml_file_path})")

                        for _ in range(target_count):
                            # 构建告警详情
                            infoItem = getInfoItem(alarmTime,image_base_code)

                            post_data["info"].append(infoItem)
                        
                        json_data = json.dumps(post_data)

                        # 发起POST请求
                        response = requests.post(report_url, data=json_data, headers={'Content-Type': 'application/json'})

                        # 检查响应状态码
                        if response.status_code == 200:
                            # 处理响应数据
                            print("上报成功(" + jpeg_file_path + "): - " + response.text)
                        else:
                            print('上报失败(' + jpeg_file_path + '):')
                            print(response.text)

# 构建告警详情
def getInfoItem(alarmTime,base64_code):
    result = { 
        "cardId":"", 
        "alarmTime":str(alarmTime), 
        "alarmPicture":base64_code
    }
    return result
    
# 例子：py .\report_event_image.py D:\工作交接\智慧工地\北京工地\视频推理\15\2023-02-08\PU_21100101_00_20230208_144407_1675839257672_helmet_xml  http://192.168.19.13:13800/report
if __name__ == "__main__":
    # jpeg文件目录
    image_dir = ""

    # 上报事件的url
    report_url = ""

    if sys.argv.__len__() > 1 and sys.argv[1] != 'null':
        image_dir = sys.argv[1]

    if sys.argv.__len__() > 2 and sys.argv[2] != 'null':    
        report_url = sys.argv[2]
        
    report_event_image(image_dir, report_url)     