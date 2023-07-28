from PIL import Image
from string import Template
import os
import sys
import requests
import json 
import base64

import ai_predict_to_xml as AI_PREDICT 

# 对比模型
TARGET_AI_TASK_ID='8f97a571-22e5-41e3-afed-8f525d8338af'
target_xml_dir = './target_xml'
# 原模型： 
COMPARE_AI_TASK_ID='e9e35528-71c7-4b62-9b88-8970bee55ab6'
compare_xml_dir = './compare_xml'
def check_ai_models(img_dir,target_ai_task_id,compare_ai_task_id):
    # emptyOutDir('./target_xml')
    print('清空目录：' + target_xml_dir)
    # emptyOutDir('./compare_xml')
    print('清空目录：' + compare_xml_dir)

    # 推理目标模型

    AI_PREDICT.setAiTaskId(target_ai_task_id)
    AI_PREDICT.ai_predict(img_dir,target_xml_dir,True)

    # 推理对比模型

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
    print("----  检测结果  ----")
    print(f'测试图片总数：{ img_count } - 全部为包含目标检测内容的测试集')
    print(f'目标模型检测推理结果 (目标模型识别/总数)： { target_xml_count }/{ img_count }   目标模型识别率：{ round(target_xml_count / img_count * 100,2) }%')
    print(f'对比模型检测推理结果 (对比模型识别/总数)： { compare_xml_count }/{ img_count}   对比模型识别率：{ round(compare_xml_count / img_count * 100,2) }%')
    print(f'相同识别结果（对比模型相同识别数/目标模型识别数）： { same_xml_count }/{ target_xml_count }')
    print(f'目标与对比模型重复识别率： { round((same_xml_count / target_xml_count)* 100,2) }%')
    print(f'指标增长个数 (新增识别/识别总数)： { target_xml_count - same_xml_count} / { target_xml_count }' )
    print(f'目标识别率提升： { round((target_xml_count - same_xml_count) / target_xml_count* 100,2) }%' )



# 清空输出目录
def emptyOutDir(out_dir): 
    if os.path.exists(out_dir):
        os.remove(out_dir)
        os.makedirs(out_dir)
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