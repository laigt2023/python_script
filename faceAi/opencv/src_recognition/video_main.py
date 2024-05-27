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
import sys
from PIL import Image, ImageDraw, ImageFont

print(onnxruntime.get_available_providers())
print(onnxruntime.get_device())
onnxruntime.set_default_logger_severity(3)

# 是否显示画框图片 True-显示/False-不显示
IS_SHOW_TARGET_IMG = True
IS_VIDEO = True
RECOGNITION_COUNT = 0

# 匹配成功阀值
COMPARISON_VALUE = 0.3
COMPARISON_LIKE_VALUE = COMPARISON_VALUE + 0.05

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)
model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)

# 目标图片缓存
TARGET_BBOXES_CACHE=np.array([])
TARGET_KPSS_CACHE=np.array([])
TARGET_FEAT_LIST_CACHE=[]

# 是否保存匹配成功的图片 True-保存/False-不保存
IS_SAVE_FACE_CP_SUCCESS_IMG=True
SAVE_FACE_CP_SUCCESS_IMG_COUNT = 0
SAVE_FACE_CP_SUCCESS_IMG_PATH='./cp_sucess_img/'

# 是否显示记录耗时
IS_SHOW_RECORDING_TIME = False

def get_img(path):
     return cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str)
    args = parser.parse_args()
    if not args.img1:
        args.img1 = '../img/face3.jpg'
    return parser.parse_args()

def funDb(target_img):
    global IS_SHOW_TARGET_IMG
    global RECOGNITION_COUNT
    global TARGET_BBOXES_CACHE
    global TARGET_KPSS_CACHE
    global IS_SAVE_FACE_CP_SUCCESS_IMG
    global SAVE_FACE_CP_SUCCESS_IMG_PATH
    global SAVE_FACE_CP_SUCCESS_IMG_COUNT
    global IS_SHOW_RECORDING_TIME
    global COMPARISON_VALUE
    global COMPARISON_LIKE_VALUE

    old_frame = target_img
    faet_db = DB.load_face_db()

    # max_num 最大识别人脸数
    if TARGET_BBOXES_CACHE.any() and TARGET_KPSS_CACHE.any() and not IS_VIDEO:
        target_bboxes = TARGET_BBOXES_CACHE
        target_kpss = TARGET_KPSS_CACHE
    else:
        TARGET_BBOXES_CACHE, TARGET_KPSS_CACHE = detector.autodetect(target_img, max_num=10)
        target_bboxes = TARGET_BBOXES_CACHE
        target_kpss = TARGET_KPSS_CACHE

    if target_bboxes.shape[0]==0:
        result = []

    # 获取到每个人脸的 feat
    feat_start_time = time.time()
    target_feat_list = []

    
    for box_num in range(len(target_bboxes)):
        # 加载对应的人脸信息
        f = rec.get(target_img, target_kpss[box_num])
        if IS_SHOW_RECORDING_TIME:
            print("feat编码耗时:", round(time.time() - feat_start_time,2),'秒', 'box数量',len(target_bboxes) )   
        TARGET_FEAT_LIST_CACHE.append(f)
        target_feat_list.append(f)    

    result=[]

    for num in range(len(target_feat_list)):
       
        for db in faet_db:
            db_persion_name = db["name"]
            face_db_feat = db["feat"]
            
            # 进行人脸数据对比        
            sim = rec.compute_sim(target_feat_list[num], face_db_feat)

            RECOGNITION_COUNT += 1
            box = target_bboxes[num]
            x1,y1,x2,y2,o=int(box[0]),int(box[1]),int(box[2]),int(box[3]),float(box[4])
            cv.rectangle(target_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
            
            if sim < COMPARISON_VALUE:
                conclu = 'NOT the same person'
            else:
                conclu = 'ARE the same person'
                result.append({"name":db_persion_name,"sim":sim,"box":[x1, y1, x2, y2]})   
                # 展示图片人脸信息
                # if IS_SHOW_TARGET_IMG: 
                #     target_img=cvAddChineseText(target_img,db_persion_name + ' '+ str(round(sim,2)), (x1, y1 - 18 ),(0,255,0),15)

                # 保存识别成果的图片
                # if IS_SAVE_FACE_CP_SUCCESS_IMG:
                #     SAVE_FACE_CP_SUCCESS_IMG_COUNT = SAVE_FACE_CP_SUCCESS_IMG_COUNT + 1 
                #     cv.imwrite(SAVE_FACE_CP_SUCCESS_IMG_PATH + str(SAVE_FACE_CP_SUCCESS_IMG_COUNT) +'.jpg',old_frame)
                #     cv.imwrite(SAVE_FACE_CP_SUCCESS_IMG_PATH + str(SAVE_FACE_CP_SUCCESS_IMG_COUNT) +'.jpeg',target_img)

                # cv.putText(target_img,db_persion_name + ' '+ str(round(sim,2)), (x1, y1 - 5 ),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)

    # 筛选出同一区域内匹配度最高的人脸信息          
    result = simUpFaces(result)

    # 显示图片时，添加人脸信息画框
    if IS_SHOW_TARGET_IMG:
        for num in range(len(result)):
            face = result[num]
            target_img=cvAddChineseText(target_img,face['name'] + ' '+ str(round(face['sim'],2)), (face['box'][0], face['box'][1] - 18 ),(0,255,0),15)


        cv.imshow('frame',target_img)

    # 每一帧都保存
    if IS_SAVE_FACE_CP_SUCCESS_IMG:
        # 添加身份信息到图片当中
        for num in range(len(result)):
            face = result[num]
            target_img=cvAddChineseText(target_img,face['name'] + ' '+ str(round(face['sim'],2)), (face['box'][0], face['box'][1] - 18 ),(0,255,0),15)

        SAVE_FACE_CP_SUCCESS_IMG_COUNT = SAVE_FACE_CP_SUCCESS_IMG_COUNT + 1 
        # 原图
        # cv.imwrite(SAVE_FACE_CP_SUCCESS_IMG_PATH + str(SAVE_FACE_CP_SUCCESS_IMG_COUNT) +'.jpg',old_frame)
        # 画框图
        cv.imwrite(SAVE_FACE_CP_SUCCESS_IMG_PATH + str(SAVE_FACE_CP_SUCCESS_IMG_COUNT) +'.jpeg',target_img)       
    return result

# 筛选出同一区域内匹配度最高的人脸信息
def simUpFaces(face_array):
    map = {}
    result = []
    for item in face_array:
        
        number_array = item['box']
        key = str(number_array[0])+ "_" + str(number_array[1])+ "_" + str(number_array[2])+ "_" + str(number_array[3])

        # 比较一下临时存储对象中的人脸数据sim值是否为最大
        if not key in map:
            map[key]=item 
        elif item['sim'] > map[key]['sim']:
            map[key]=item 
        
    # 再次封装成返回数据
    for item in map:
        result.append(map[item])
    return result      


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
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


if __name__ == '__main__':
    # cap = cv.VideoCapture('../img/PU_01.mp4')
    # cap = cv.VideoCapture(r'E:\faces_emore\dataset2\fa20230919_100452.mp4')
    # cap = cv.VideoCapture(r'./video/test02.mp4')
    # cap = cv.VideoCapture(r'./video/Tai_yuan_01.mp4')
    # cap = cv.VideoCapture(r'./video/PU_03.mp4')
    # 2k
    #cap = cv.VideoCapture(r'../data/video/face_test.mp4')
    # 1080p
    # cap = cv.VideoCapture(r'../data/video/00005.mp4')
    cap = cv.VideoCapture(r"rtsp://192.168.19.202/1-2")
    count = 0
    fps = 1

    if(sys.argv.__len__() > 1):

        if 'false' == str(sys.argv[1]):
            print(sys.argv[1] , '不显示视频内容')
            IS_SHOW_TARGET_IMG = False

    if not cap.isOpened():
        print("Error opening video file")
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
       
           
        if count % fps == 0:
            start_time = time.time()
            result = funDb(frame)
            
            if len(result) > 0:
                print("同一帧:")
            
            for one in result:
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'名称',one['name'],'sim',str(round(one['sim'],2)),'box',one['box'],"当前帧耗时:", round(time.time() -start_time,2),'秒')
            
            # cv.imshow('frame',frame)
            # 导出图片  需要手动创建./input_frame 文件夹
            # cv.imwrite('./input_frame/img_'+ str(count) +'.jpg', frame)
            if IS_SHOW_RECORDING_TIME:
                print("当前帧耗时:", round(time.time() -start_time,2),'秒' )      

        # cv.imshow('Video', frame)
        count = count + 1
        if IS_SHOW_TARGET_IMG:
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    if IS_SHOW_TARGET_IMG:
        cv.destroyAllWindows()