
import os
import face_recognition
import cv2 as cv
import datetime
import time
import face_save_db as DB

# 不打印日志
LOGO_SHOW=False

# 人脸库全局变量
FACE_ENCODES=[]

# 是否使用本地文件加载FACE_ENCODES 人脸库
USE_SAVE_FACE_ENCODES_FILE = True

# 人脸库文件目录
FACE_ENCODES_DB_FILE = DB.FACE_ENCODES_DB_FILE

# 人脸比对次数
COMPARISON_COUNT=0

# 当前人脸匹配信息（缓存）
TEST_FACE_CACHE={
    'image':None,
    'locations':None,
    'encodings': None
}

def test():
    # 加载图片
    zms = face_recognition.load_image_file("./face1.jpg")
    zms = face_recognition.load_image_file("./face1.jpg")

    # 人脸检测
    zms_fcae = face_recognition.face_locations(zms)

    # 人脸特征编码
    zms_encoding= face_recognition.face_encodings(zms,zms_fcae)
    for one in zms_encoding:
        Log('one',one)

    # 把所有人脸放在一起，当做数据库
    encodings = [zms_encoding]
    names = ["zms de hua"]

# 加载人脸库
def loadFaceDb(face_db_dir):
    global FACE_ENCODES
    global USE_SAVE_FACE_ENCODES_FILE
    # 初始化人脸库
    FACE_ENCODES = []

    if USE_SAVE_FACE_ENCODES_FILE and os.path.exists(FACE_ENCODES_DB_FILE):
        print("加载人脸库文件 (使用JSON文件加载): ",FACE_ENCODES_DB_FILE)
        FACE_ENCODES = DB.load_face_db()    

    if len(FACE_ENCODES) > 0:
        return FACE_ENCODES
    
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

                item = {
                        'name':one_name,
                        'face_encode':face_one_encoding,
                        'locations': face_one_locations
                        }
                
                FACE_ENCODES.append(item)


    return FACE_ENCODES            

# 初始化缓存
def init_face_cache():
    global TEST_FACE_CACHE
    TEST_FACE_CACHE['image'] = None
    TEST_FACE_CACHE['locations'] = None
    TEST_FACE_CACHE['encodings'] = None
   
# face_encode 人脸库中的人脸编码   two_pic：检测人脸图片   （注：检测人脸图片可能存在多张人脸信息）
def demoFunc(face_encode,two_pic="./face1.jpg",tolerance=0.45):
    
    global TEST_FACE_CACHE
    global COMPARISON_COUNT

    # 目标图像
    # if TEST_FACE_CACHE and TEST_FACE_CACHE['image'] != None:
    #     face_2 = TEST_FACE_CACHE['image']
    # else:
    #     TEST_FACE_CACHE['image'] = face_recognition.load_image_file(two_pic)
    #     face_2 = TEST_FACE_CACHE['image']
    face_2 = face_recognition.load_image_file(two_pic)

    if TEST_FACE_CACHE and TEST_FACE_CACHE['locations'] != None:
        face_2_locations = TEST_FACE_CACHE['locations']
    else:
        TEST_FACE_CACHE['locations'] = face_recognition.face_locations(face_2)
        face_2_locations = TEST_FACE_CACHE['locations']

    if TEST_FACE_CACHE and TEST_FACE_CACHE['encodings'] != None:
        face_2_encodings =TEST_FACE_CACHE['encodings']
    else:
        TEST_FACE_CACHE['encodings'] = face_recognition.face_encodings(face_2, known_face_locations=face_2_locations, num_jitters=1)
        face_2_encodings = TEST_FACE_CACHE['encodings']
        

    array_face_locations = []
    for  i in range(len(face_2_encodings)):
        
        current_encode =face_2_encodings[i]
        Log("compare_faces - 人脸数据匹配中")
        # tolerance 越低，匹配度数要求越高 
        # tolerance – 将人脸之间的距离视为匹配。越低越严格。0.6 是典型的最佳值。
    
        result = face_recognition.compare_faces([face_encode],current_encode,tolerance)
        COMPARISON_COUNT = COMPARISON_COUNT + 1

        if result[0]:
            array_face_locations.append({"location":face_2_locations[i]})

       
    return array_face_locations

# 统一打印日志
def Log(msg,result=None):
    global LOGO_SHOW

    if LOGO_SHOW:
        print(msg,result)

# image 图片
# face_loaction 人脸位置
def drawFaceFunc(image,face_loaction,text='unknow'):
    y0, x1 ,y1 ,x0 = face_loaction
    cv.rectangle(image,pt1=(x0,y0),pt2=(x1,y1),color=(0,0,255),thickness=3)
    cv.putText(image,text, (x0 + 5, y0 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def showFace(pic_path=''):
    img = face_recognition.load_image_file(pic_path)
    face_landmarks_list = face_recognition.face_landmarks(img)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            Log(face_landmarks[facial_feature])

# 人脸比对方法
# picDir ： 人脸库目录  例:'./data/jm/'
# test_pic ： 需要进行人脸匹配的图片路径  例:'./face3.jpg'
def faceDemo(picDir='',test_pic=''):
    global TEST_FACE_CACHE
    global COMPARISON_COUNT

    face_db_encodes=[]
  
    # 展示图片
    image = face_recognition.load_image_file(test_pic)

    # 匹配文件计数器
    global COMPARISON_COUNT
    COMPARISON_COUNT = 0

    face_map={}
    # 对比所有人脸库
    print("人脸库加载中,请稍后...")
    # 记录加载人脸库耗时记录
    load_faces_start_time = time.time()
    face_db_encodes = loadFaceDb(picDir)
    load_faces_end_time = time.time()
  
    
    # 初始化缓存
    init_face_cache()
    print("人脸库匹配中,请稍后...")

    # 记录加载人脸库匹配耗时记录
    comparison_start_time = time.time()

    for face_db_one in face_db_encodes:
        # print(face_db_one['face_encode'])
        # print(type(face_db_one['face_encode']))
        # print(face_db_one['face_encode'].__class__)

        locations = demoFunc(face_db_one['face_encode'],test_pic)
        if locations:
            print("已匹配: this Person is", face_db_one['name'])
            for face in locations:
                face['persion_name'] = face_db_one['name']

                face_key = face['location']    

                # 封装人脸定位与匹配的人脸名称 ：  { location:[ {location,persion_name} ] }  
                if face_key in face_map:
                    face_map[face_key].append(face)
                else:
                    face_map[face_key] = [face]    

    comparison_end_time = time.time()

    # 遍历获取到的人脸匹配信息        { location:[ {location,persion_name} ] }  
    for face_key in face_map:
        list = face_map[face_key]
        persion_text = ''
        if len(list) > 1:
            for face_one in list:
                persion_text = persion_text + '/ ' + face_one['persion_name']

            persion_text = persion_text.lstrip('/ ')    
        else:
            persion_text = list[0]['persion_name']

        
        image=drawFaceFunc(image,list[0]['location'],persion_text)    
        print(list[0]['location'],persion_text)

    # 显示人脸库信息
    print("人脸库：", len(face_db_encodes))
    print("加载人脸库耗时", round(load_faces_end_time - load_faces_start_time,2), "秒")
    if TEST_FACE_CACHE and TEST_FACE_CACHE['encodings']:
        print("目标图像可识别人脸数", len(TEST_FACE_CACHE['encodings']))
    else:
        print("目标图像可识别人脸数", 0)

    print("人脸比对次数", COMPARISON_COUNT) 
    print("人脸库匹配耗时", round(comparison_end_time - comparison_start_time,2), "秒")    
    return image 

if __name__ == "__main__":
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("程序开始时间：", current_time)
    start_time = time.time()

    # 人脸匹配
    image = faceDemo(picDir='./data/jm/',test_pic='./face3.jpg')
    # image = faceDemo(picDir='./data/jm/',test_pic='./face1.jpg')

    end_time =  time.time()
    run_time = end_time - start_time
    # 打印运行时间

    print("程序开始时间：", current_time)
    print("程序当前时间：", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("程序总耗时：", round(run_time,2), "秒")

    cv.imshow("face",image)
    if cv.waitKey(0) & 0xFF ==("q"):
        cv.destroyAllWindows()  

    
