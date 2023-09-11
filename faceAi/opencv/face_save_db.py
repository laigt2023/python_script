import os
import face_recognition
import datetime
import time
import json
import numpy as np

# 人脸库文件目录
FACE_ENCODES_DB_FILE = './face_db.json'

# 人脸图片存放目录
FACE_ENCODES_DB_IMAGE_DIR = './data/jm/'

# 解析人脸库文件目录，并生成人脸库JSON文件
def save_face_db(face_db_dir):
    global FACE_ENCODES_DB_FILE

    face_one_encodings_db = []

    for root, dirs, files in os.walk(face_db_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jepg') or file.endswith('.png'):
                one_name = file.split(".")[1]

                face_one = face_recognition.load_image_file(face_db_dir + file)

                # face_recognition.face_locations(img,number_of_times_to_upsample=1,model='hog')
                # 返回图像中每张人脸的人脸特征位置列表；
                # number_of_times_to_upsample – 对图像进行多少次上采样以查找人脸。数字越大，人脸越小；
                # model – "hog"不太准确，但在CPU上更快。"cnn"是GPU / CUDA加速的一个更准确的深度学习模型。

                face_one_locations = face_recognition.face_locations(face_one)

                # face_recognition.face_encodings(face_image, known_face_locations=None, num_jitters=1, model='small')
                # 返回图像中每张人脸的 128 维人脸编码。
                # known_face_locations - 可选 - 每个面孔的边界框（如果已经知道它们）- face_locations。
                # num_jitters – 计算编码时重新采样人脸的次数。越高越准确，但速度越慢（即 100 表示慢 100 倍）。
                # model – “large” (默认) 或 “small”仅返回5个点，但速度更快。
                face_one_encoding = face_recognition.face_encodings(face_one, known_face_locations=face_one_locations, num_jitters=1)[0]

                encodings_json = json.dumps(face_one_encoding.tolist())    
   

                # itemStr = "{"+f"'name':{one_name},'face_encode':{face_one_encoding},'locations':{face_one_locations}" + "}"
                item = {
                    'name':one_name,
                    'face_encode':encodings_json,
                    'locations':face_one_locations
                }
                face_one_encodings_db.append(item)
                
                
    
    # 写入文件
    with open(FACE_ENCODES_DB_FILE, 'w', encoding='utf-8') as xml_file:
        json_str = json.dumps(face_one_encodings_db)
        xml_file.write(json_str)         

# 加载人脸库
def load_face_db():
    # 读取文件
    with open(FACE_ENCODES_DB_FILE, 'r') as f:
        # 读取JSON内容
        face_encodings_db = json.load(f)

        for one in face_encodings_db:
            encodings_list = one['face_encode'].lstrip("[").rstrip("]").split(", ")
            encodings = []
            for encodingsin in encodings_list:
                encodings.append(float(encodingsin))
    
            one['face_encode'] = encodings
        return face_encodings_db


if __name__ == "__main__":
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("人脸数据库加载开始时间：", current_time)
    start_time = time.time()

    # 写入文件
    save_face_db(FACE_ENCODES_DB_IMAGE_DIR)

    # 读取文件
    # load_face_db()

    end_time =  time.time()
    run_time = end_time - start_time
    # 打印运行时间

    print("人脸数据库加载开始时间：", current_time)
    print("人脸数据库加载完成时间：", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("人脸数据库写入总耗时：", round(run_time,2), "秒")




