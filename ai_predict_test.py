from PIL import Image
from string import Template
import os
import sys
import requests
import json 
import base64
import datetime

import ai_predict_to_xml as AI_PREDICT 

# 用于进行AI模型对比测试

# 现在的时间
startTime = datetime.datetime.now()


# 对比模型
TARGET_AI_TASK_ID='8f97a571-22e5-41e3-afed-8f525d8338af'
target_xml_dir = './target_xml'
# 原模型: 
COMPARE_AI_TASK_ID='e9e35528-71c7-4b62-9b88-8970bee55ab6'
compare_xml_dir = './compare_xml'

# 跳过推理，直接输出上一次结果  True-跳过  False-不跳过推理
SKIP_PREDICT=False

def setSkipPredict(skip_predict):
    global SKIP_PREDICT
    SKIP_PREDICT = skip_predict

def check_ai_models(img_dir,target_ai_task_id,compare_ai_task_id):
    global startTime
        
    emptyOutDir('./target_xml')
    print('清空目录:' + target_xml_dir)
    emptyOutDir('./compare_xml')
    print('清空目录:' + compare_xml_dir)

    if SKIP_PREDICT != True:
        print('跳过推理目标模型')
        # 推理目标模型
        AI_PREDICT.setAiTaskId(target_ai_task_id)
        AI_PREDICT.ai_predict(img_dir,target_xml_dir,True)

        # 推理对比模型
        if compare_ai_task_id != '':
            AI_PREDICT.setAiTaskId(compare_ai_task_id)
            AI_PREDICT.ai_predict(img_dir,compare_xml_dir,True)

    
    # 输入文件总数
    img_count = 0
    # 目标识别结果
    target_xml_count = 0
    # 相同识别结果
    same_xml_count = 0

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png') :
                img_count += 1

    for root, dirs, files in os.walk(target_xml_dir):
        for file in files:
            if file.endswith('.xml'):
                target_xml_count += 1
                if os.path.exists(compare_xml_dir + os.path.sep + file):
                    same_xml_count += 1

    compare_xml_count = 0
    for root, dirs, files in os.walk(compare_xml_dir):
        for file in files:
            if file.endswith('.xml'):
                compare_xml_count += 1

    endTime = datetime.datetime.now()

    result = f"图片路径: {img_dir}\n"
    result += f"----  检测结果 开始时间:{ startTime.strftime('%Y-%m-%d %H:%M:%S') } ----\n"
    result += f'测试图片总数:{ img_count } - 全部为包含目标检测内容的测试集\n'
    result += f'目标模型ID: { target_ai_task_id }\n'
    result += f'对比模型ID: { compare_ai_task_id }\n'
    result += f'目标模型识别结果 (目标模型识别/图片总数): { target_xml_count }/{ img_count }   目标模型识别率:{ numDividedFormat(target_xml_count , img_count) }\n'
    result += f'对比模型识别结果 (对比模型识别/图片总数): { compare_xml_count }/{ img_count}   对比模型识别率:{ numDividedFormat(compare_xml_count , img_count) }\n'
    result += f'二次相同识别结果(相同识别数/目标模型识别数): { same_xml_count }/{ target_xml_count }\n'
    result += f'二次识别率(再次识别率): { numDividedFormat(same_xml_count , compare_xml_count) }\n'
    result += f'模板模型识别增长数 (新增识别个数/识别总数): { target_xml_count - same_xml_count} / { target_xml_count }\n'
    result += f'目标识别率提升: { numDividedFormat((target_xml_count - same_xml_count) , same_xml_count) }\n'
    result += f"----  检测结果 结束时间: { endTime.strftime('%Y-%m-%d %H:%M:%S') }  ----\n"
    print(result)

    # 写入文件  w-重新 a-追加
    # with open("./result.log", 'w', encoding='utf-8') as txt_file:
    with open("./result.log", 'a', encoding='utf-8') as txt_file:
        result += f"\r\n"
        txt_file.write(result)
        print("对比结果文件生成成功(./result.log)")

    return
# 除法，分母为0时返回0
def numDividedFormat(a,b):
    if b == 0:
        return '0%'
    return f'{round(a/b * 100,2)}%'

# 清空输出目录
def emptyOutDir(out_dir): 
    if os.path.exists(out_dir):
        # os.remove(out_dir)
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                os.remove(os.path.join(root, file))
    else:
        os.makedirs(out_dir)

# 主函数
if __name__ == "__main__":
    # 输入校验
    if(sys.argv.__len__() < 2):
        printMsg('请输入参数1: 图片目录')
        exit() 

    # 推理图片目录
    img_dir = sys.argv[1]

    if sys.argv.__len__() > 2 and sys.argv[2] != 'null':
        TARGET_AI_TASK_ID = sys.argv[2]

    if sys.argv.__len__() > 3 and sys.argv[3]!= 'null':    
        COMPARE_AI_TASK_ID = sys.argv[3] 

    check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID)