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

print(onnxruntime.get_available_providers())
print(onnxruntime.get_device())
onnxruntime.set_default_logger_severity(3)

IS_SHOW_TARGET_IMG = True
RECOGNITION_COUNT = 0

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

def get_img(path):
     return cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str)
    args = parser.parse_args()
    if not args.img1:
        args.img1 = '../img/face3.jpg'
    return parser.parse_args()


def func(target_img,fac_db_img_path):
    global IS_SHOW_TARGET_IMG
    global RECOGNITION_COUNT
    global TARGET_BBOXES_CACHE
    global TARGET_KPSS_CACHE

    # 加载人脸库信息
    load_face_start_time = time.time()
    
    fac_db_array = fac_db_img_path.split("/")
    filename = fac_db_array[len(fac_db_array) -1]
    db_persion_name = filename.split(".")[1]

    fac_db_img = get_img(fac_db_img_path)
    fac_db_bboxes, fac_db_kpss = detector.autodetect(fac_db_img, max_num=1)
    if fac_db_bboxes.shape[0]==0:
        return -1.0, "Face not found in Image-2"

    face_db_kps = fac_db_kpss[0]
    face_db_feat = rec.get(fac_db_img, face_db_kps)
    print("加载人脸耗时", round(time.time() - load_face_start_time,2) ,'秒')

    # max_num 最大识别人脸数
    if TARGET_BBOXES_CACHE.any() and TARGET_KPSS_CACHE.any() :
        target_bboxes = TARGET_BBOXES_CACHE
        target_kpss = TARGET_KPSS_CACHE
    else:
        TARGET_BBOXES_CACHE, TARGET_KPSS_CACHE = detector.autodetect(target_img, max_num=10)
        target_bboxes = TARGET_BBOXES_CACHE
        target_kpss = TARGET_KPSS_CACHE

    if target_bboxes.shape[0]==0:
        return -1.0, "Face not found in Image-1"
    
    # 多张人脸
    for num in range(len(target_bboxes)):
        # 目标图片中的人脸数据
        target_feat_list = []
        # 缓存中没有人脸信息，则按照下标加载
        if len(TARGET_FEAT_LIST_CACHE):
            target_feat_list = TARGET_FEAT_LIST_CACHE
        else:
            for box_num in range(len(target_bboxes)):
                # 加载对应的人脸信息
                f = rec.get(target_img, target_kpss[box_num])
                TARGET_FEAT_LIST_CACHE.append(f)
                target_feat_list.append(f)

        # 进行人脸数据对比        
        sim = rec.compute_sim(target_feat_list[num], face_db_feat)

        RECOGNITION_COUNT += 1
        if sim<0.2:
            conclu = 'They are NOT the same person'
        elif sim>=0.2 and sim<0.28:
            conclu = 'They are LIKELY TO be the same person'
        else:
            conclu = 'They ARE the same person'
            if IS_SHOW_TARGET_IMG: 
                box = target_bboxes[num]
                x1,y1,x2,y2,o=int(box[0]),int(box[1]),int(box[2]),int(box[3]),float(box[4])
                cv.rectangle(target_img, (x1, y1), (x2, y2), (0, 0, 255), 2)   
                cv.putText(target_img,db_persion_name + " " + str(round(sim,2)), (x1, y1 - 5 ),cv.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

        print(db_persion_name, sim)            
    return sim, conclu

def funDb(target_img):
    global IS_SHOW_TARGET_IMG
    global RECOGNITION_COUNT
    global TARGET_BBOXES_CACHE
    global TARGET_KPSS_CACHE

    faet_db = DB.load_face_db()

    # max_num 最大识别人脸数
    if TARGET_BBOXES_CACHE.any() and TARGET_KPSS_CACHE.any():
        target_bboxes = TARGET_BBOXES_CACHE
        target_kpss = TARGET_KPSS_CACHE
    else:
        TARGET_BBOXES_CACHE, TARGET_KPSS_CACHE = detector.autodetect(target_img, max_num=10)
        target_bboxes = TARGET_BBOXES_CACHE
        target_kpss = TARGET_KPSS_CACHE

    if target_bboxes.shape[0]==0:
        return []

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

    for num in range(len(target_feat_list)):
       
        for db in faet_db:
            db_persion_name = db["name"]
            face_db_feat = db["feat"]
            
            # 进行人脸数据对比        
            sim = rec.compute_sim(target_feat_list[num], face_db_feat)

            RECOGNITION_COUNT += 1
            box = target_bboxes[num]
            x1,y1,x2,y2,o=int(box[0]),int(box[1]),int(box[2]),int(box[3]),float(box[4])

            if IS_SHOW_TARGET_IMG: 
                cv.rectangle(target_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
            
            if sim<0.2:
                conclu = 'NOT the same person'
            elif sim>=0.2 and sim<0.28:
                conclu = 'LIKELY TO be the same person'
            else:
                conclu = 'ARE the same person'
                result.append({"name":db_persion_name,"sim":sim,"box":[(x1, y1), (x2, y2)]})   
                if IS_SHOW_TARGET_IMG: 
                    cv.putText(target_img,db_persion_name + ' '+ str(round(sim,2)), (x1, y1 - 5 ),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
       
    return result


if __name__ == '__main__':
    RECOGNITION_COUNT = 0
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("程序开始时间：", current_time)
    start_time = time.time()

    args = parse_args()
    target_img = get_img(args.img1)
    
    # face_db_dir = '../data/jm/'
    # for root, dirs, files in os.walk(face_db_dir):
    #         for file in files:
    #             if file.endswith('.jpg') or file.endswith('.jepg') or file.endswith('.png'):
    #                 fac_db_img_path = face_db_dir + file
    #                 output = func(target_img,fac_db_img_path)
    

    result = funDb(target_img)
    if len(result):
        print("人脸比对结果:")
    else:
        print("人脸库不存在匹配信息")

    for one in result:
        print(one)

    run_time = time.time() - start_time

    print("人脸比对次数", RECOGNITION_COUNT,"每次平均耗时", round(run_time / RECOGNITION_COUNT,2),"秒")
    print("程序开始时间：", current_time)
    print("程序当前时间：", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("程序总耗时：", round(run_time,2), "秒")

    if IS_SHOW_TARGET_IMG:      
        cv.imshow("insightface",target_img)
        if cv.waitKey(0) & 0xFF ==("q"):
            cv.destroyAllWindows() 

