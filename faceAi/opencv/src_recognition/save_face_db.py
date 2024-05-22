import os
import cv2 as cv
import datetime
import time
import json
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
import os.path as osp
import sys

# 人脸库文件目录
# FACE_ENCODES_DB_FILE = './fzx_face_db.json'
FACE_ENCODES_DB_FILE = './LongKou_face_db.json'
# 人脸图片存放目录 (如：../data/jm/db/)
# FACE_ENCODES_DB_IMAGE_DIR = '../data/jm/fzx_face_db/'
FACE_ENCODES_DB_IMAGE_DIR =r"D:\工作交接\穗建人脸考勤\龙口人脸库\d1e394d934ff4db096c3cd681637e432/"
# FACE_ENCODES_DB_IMAGE_DIR =r"D:\工作交接\穗建人脸考勤\龙口人脸库\test/"

if not os.path.exists(FACE_ENCODES_DB_FILE):
    with open(FACE_ENCODES_DB_FILE, 'w'):
        pass

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_img(path):
     return cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)

# 解析人脸库文件目录，并生成人脸库JSON文件
def save_face_db(face_db_dir,jsonFileName=None):
    global FACE_ENCODES_DB_FILE
    global app

    if not os.path.exists(FACE_ENCODES_DB_FILE):
        with open(FACE_ENCODES_DB_FILE, 'w'):
            pass

    face_one_encodings_db = []

    assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')
    detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
    detector.prepare(0)
    model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
    rec = ArcFaceONNX(model_path)
    rec.prepare(0)

    count_num = 0
    success_num = 0
    error_num = 0 
    error_array=[]
   
    for root, dirs, files in os.walk(face_db_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jepg') or file.endswith('.png'):
                print("加载人脸文件:" + file)
                count_num = count_num + 1 
                try:
                    # 加载人脸库信息
                    load_face_start_time = time.time()
                    
                    filename = file
                    db_persion_name = filename.split(".")[0]

                    fac_db_img = get_img(os.path.abspath(os.path.join(root, file)))
                    fac_db_bboxes, fac_db_kpss = detector.autodetect(fac_db_img, max_num=1)
                    if fac_db_bboxes.shape[0]==0:
                        error_num = error_num + 1
                        error_array.append(filename)
                        continue

                    face_db_kps = fac_db_kpss[0]
                    face_db_feat = rec.get(fac_db_img, face_db_kps)
                
                    print("加载人脸耗时(",filename," : ",db_persion_name,")", round(time.time() - load_face_start_time,2) ,'秒')
                    item = {
                        "name":db_persion_name,
                        "feat":face_db_feat.tolist()
                    }
                    face_one_encodings_db.append(item)
                    success_num = success_num + 1
                except:
                    error_num = error_num + 1
                    error_array.append(filename)
                    continue
        
   
    print(f"总文件：{count_num} 个，成功编码：{success_num} 个,异常编码{error_num} 个")
    if len(error_array) > 0:
        print("人脸信息编码异常文件:")
        for error_filename in error_array:
            print(error_filename)

    # 写入文件
    saveFile = FACE_ENCODES_DB_FILE

    if jsonFileName:
        saveFile=jsonFileName
        
    with open(saveFile, 'w', encoding='utf-8') as xml_file:
        json_str = json.dumps(face_one_encodings_db)
        xml_file.write(json_str)         

    return face_one_encodings_db 
# 加载人脸库
def load_face_db():
    if not os.path.exists(FACE_ENCODES_DB_FILE):
        with open(FACE_ENCODES_DB_FILE, 'w'):
            return []
    # 读取文件
    with open(FACE_ENCODES_DB_FILE, 'r') as f:
        # 读取JSON内容
        face_encodings_db = json.load(f)

        for one in face_encodings_db:
            # encodings_list = one['face_encode'].lstrip("[").rstrip("]").split(", ")
            # encodings = []
            # for encodingsin in encodings_list:
            #     encodings.append(float(encodingsin))
            one['feat'] = np.array(one['feat'] )
        return face_encodings_db

# 重建人脸库  face_dir - 人脸库目录 不填为默认路径
def face_db_rebuild(face_dir):
    print("face_dir：",face_dir)
    if face_dir:
        return save_face_db(face_dir)
    else:
        return save_face_db(FACE_ENCODES_DB_IMAGE_DIR)

if __name__ == "__main__":
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("人脸数据库加载开始时间：", current_time)

    start_time = time.time()

    if sys.argv.__len__() > 2:
      jsonFileName = sys.argv[1]
      save_face_db(FACE_ENCODES_DB_IMAGE_DIR,jsonFileName)
    else:  
      # 写入文件
      save_face_db(FACE_ENCODES_DB_IMAGE_DIR)

    # 读取文件
    # load_face_db()

    end_time =  time.time()
    run_time = end_time - start_time
    # 打印运行时间

    print("人脸数据库加载开始时间：", current_time)
    print("人脸数据库加载目录：", FACE_ENCODES_DB_IMAGE_DIR)
    print("人脸数据库输出文件：", FACE_ENCODES_DB_FILE)
    print("人脸数据库加载完成时间：", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("人脸数据库数据更新完成，总耗时：", round(run_time,2), "秒")




