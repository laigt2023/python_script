#!/usr/bin/env python

import os
import os.path as osp
import argparse
import cv2 as cv
import numpy as np
# import torch
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
import save_face_db as DB
import datetime
import time
from PIL import Image, ImageDraw, ImageFont
import base64
import io

onnxruntime.set_default_logger_severity(3)

# 匹配成功阀值
COMPARISON_VALUE = 0.35

# 是否开启 记录相似人脸信息  True-启动 False-关闭
IS_COMPARISON_LIKE_FACE = True
# 相似人脸阈值
LIKE_FACE_MIN_VALUE = 0.1


# 是否展示推理后图片
IS_SHOW_TARGET_IMG = False
# 人脸匹配计数器
RECOGNITION_COUNT = 0

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)
model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)

#人脸库数据
FEAT_DB = []

# 目标图片缓存  TARGET_BBOXES_CACHE  TARGET_KPSS_CACHE 是坐标信息，图片更新时需要置空此2个缓存
TARGET_BBOXES_CACHE=np.array([])
TARGET_KPSS_CACHE=np.array([])
TARGET_FEAT_LIST_CACHE=[]

def get_img(path):
     return cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str)
    args = parser.parse_args()
    if not args.img1:
        args.img1 = '../img/03.jpg'
    return parser.parse_args()

# 清空缓存，每次新图片进来时都需要清空此缓存
def clearTargetImgCache():
    global TARGET_BBOXES_CACHE
    global TARGET_KPSS_CACHE
    
    # 每次新图片进来时，都要清空缓存
    TARGET_BBOXES_CACHE = np.array([])
    TARGET_KPSS_CACHE = np.array([])

# 获取人脸库方法
def getFaceDb():
    global FEAT_DB
    
    if len(FEAT_DB) <= 0 :
        FEAT_DB = reloadFaceDb()

    return FEAT_DB

# 重新加载人脸库数据
def reloadFaceDb():
    global FEAT_DB
    FEAT_DB = DB.load_face_db()
    return FEAT_DB

# 触发重新构建人脸
def rebuildFaceDb(face_dir):
    global FEAT_DB
    DB.face_db_rebuild(face_dir)
    FEAT_DB = reloadFaceDb()
    return FEAT_DB

# 从人脸库中匹配人脸数据
def funDb(target_img,comparison_value = None):
    global IS_SHOW_TARGET_IMG
    global RECOGNITION_COUNT
    global TARGET_BBOXES_CACHE
    global TARGET_KPSS_CACHE
    global COMPARISON_VALUE

    if not comparison_value:
        comparison_value = COMPARISON_VALUE

    # 加载人脸库
    faet_db = getFaceDb()

    # max_num 最大识别人脸数
    if TARGET_BBOXES_CACHE.any() and TARGET_KPSS_CACHE.any():
        target_bboxes = TARGET_BBOXES_CACHE
        target_kpss = TARGET_KPSS_CACHE
    else:
        # 获取到每个人脸的 feat
        autodetect_start_time = time.time()
        TARGET_BBOXES_CACHE, TARGET_KPSS_CACHE = detector.autodetect(target_img, max_num=10)
        print("检测耗时:", round(time.time() - autodetect_start_time,2),'秒', "共检测到 " + str(len(TARGET_BBOXES_CACHE)) +" 张人脸" )   
        target_bboxes = TARGET_BBOXES_CACHE
        target_kpss = TARGET_KPSS_CACHE

    if target_bboxes.shape[0]==0:
        return [],target_img

   # 获取到每个人脸的 feat
    feat_start_time = time.time() 
    target_feat_list = []
    
    for box_num in range(len(target_bboxes)):
        # 加载对应的人脸信息
        f = rec.get(target_img, target_kpss[box_num])
        print("feat编码耗时:", round(time.time() - feat_start_time,2),'秒', 'box数量',len(target_bboxes) )   
        TARGET_FEAT_LIST_CACHE.append(f)
        target_feat_list.append(f)    

    result=[]
    
    compute_start_time = time.time()
    for num in range(len(target_feat_list)):
        box = target_bboxes[num]
        x1,y1,x2,y2,o=int(box[0]),int(box[1]),int(box[2]),int(box[3]),float(box[4])
        
        for db in faet_db:
            db_persion_name = db["name"]
            face_db_feat = db["feat"]
            
           
            # 进行人脸数据对比        
            sim = rec.compute_sim(target_feat_list[num], face_db_feat)

            RECOGNITION_COUNT += 1

            if IS_SHOW_TARGET_IMG: 
                cv.rectangle(target_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
            
            if sim < comparison_value:
                conclu = 'NOT the same person'
            else:
                conclu = 'ARE the same person'
                result.append({"name":db_persion_name,"sim":sim,"box":[x1,y1,x2,y2]})   
              
                if IS_SHOW_TARGET_IMG: 
                    # cv.putText(target_img,db_persion_name + ' '+ str(round(sim,2)), (x1, y1 - 5 ),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                    target_img=cvAddChineseText(target_img,db_persion_name + ' '+ str(round(sim,4)), (x1, y1 - 22 ),(0,255,0),20)

    print("feat匹配耗时:", round(time.time() - compute_start_time,2),'秒')      
    return result,target_img


# 从数据库中匹配人脸并筛选出最高匹配度的身份信息
# img 模板目标图片
# comparison_value 最低匹配值
def faceCpByDB(target_img,comparison_value = None):
    # 清除当前图片帧的编码数据
    clearTargetImgCache()
    
    if comparison_value == None:
        comparison_value = COMPARISON_VALUE

    if IS_COMPARISON_LIKE_FACE == True:
        array,target_img = funDb(target_img,0.01)
    else:
        array,target_img = funDb(target_img,LIKE_FACE_MIN_VALUE)   

    # return array,target_img
    map = {}
    result = []
    for item in array:
        
        number_array = item['box']
        key = str(number_array[0])+ "_" + str(number_array[1])+ "_" + str(number_array[2])+ "_" + str(number_array[3])

        # 比较一下临时存储对象中的人脸数据sim值是否为最大
        if not key in map:
            map[key]=item 
        elif item['sim'] > map[key]['sim']:
            map[key]=item 
    
    
 
    # 再次封装成返回数据
    for item in map:
        if map[item]['sim'] >= comparison_value:
            result.append(map[item])

    if IS_COMPARISON_LIKE_FACE == True:
        originalResult = []
        for item in map:
            if map[item]['sim'] >= LIKE_FACE_MIN_VALUE:
                originalResult.append(map[item])
        logOriginalMesg(originalResult) 

    return result,target_img      

# 中文标签
def cvAddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_BGR2RGB)

# base64转图片
def base64_to_image(encoded_string):
    # decoded_bytes = base64.b64decode(encoded_string)
    # image = Image.open(BytesIO(decoded_bytes))
    img_b64decode = base64.b64decode(encoded_string)  # base64解码
 
    image = io.BytesIO(img_b64decode)
    img = Image.open(image)
    image = img
    return cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB)

# 统一日志输出
def logOriginalMesg(msg):
    # 获取当前路径
    today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir=current_path + os.path.sep + "face_logs"
    log_file_path = f"{dir + os.path.sep + today}_face_original.log"
    # 创建日志文件夹
    if(os.path.exists(dir) == False):
        os.mkdir(dir)

    # 写入日志文件  w-重新 a-追加
    with open(log_file_path, 'a', encoding='utf-8') as txt_file:
        result = f"{current_time} original 共识别{len(msg)} 张人脸: {msg}"
        txt_file.write(f"{result}\n")
        print(result)

if __name__ == '__main__':
    RECOGNITION_COUNT = 0
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("程序开始时间：", current_time)
    start_time = time.time()

    args = parse_args()
    target_img = get_img(args.img1)
    
    # with open(args.img1, "rb") as image_file:                           
    #     encoded_string = base64.b64encode(image_file.read())
    #     target_img=base64_to_image(bytes.decode(encoded_string))
    #     # 写入日志文件  w-重新 a-追加
    #     with open("./base64.text", 'w', encoding='utf-8') as txt_file:
    #         txt_file.write(bytes.decode(encoded_string))


    result,target_img = faceCpByDB(target_img)

    
    if len(result):
        print("人脸比对结果:")
    else:
        print("人脸库不存在匹配信息")

    for one in result:
        print(one)

    run_time = time.time() - start_time

    print("执行提供程序：",onnxruntime.get_available_providers())
    print(onnxruntime.get_device())
    
    print("人脸比对次数", RECOGNITION_COUNT,"每次平均耗时", round(run_time / RECOGNITION_COUNT,4),"秒")
    print("程序开始时间：", current_time)
    print("程序当前时间：", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("程序总耗时：", round(run_time,4), "秒")

    if IS_SHOW_TARGET_IMG:      
        cv.imshow("insightface",target_img)
        if cv.waitKey(0) & 0xFF ==("q"):
            cv.destroyAllWindows() 

