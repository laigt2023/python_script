
import os
import face_recognition
import datetime
import time

# 不打印日志
LOGO_SHOW=False

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

# ope_pic 人脸库   two_pic：检测人脸图片   （注：检测人脸图片可能存在多张人脸信息）
def demoFunc(one_pic="./data/jm/2.zms.jpg",two_pic="./face1.jpg",two_pic_locations=None):
    Log("load_image_file - 人脸数据读取中")
    face_1 = face_recognition.load_image_file(one_pic)
    face_2 = face_recognition.load_image_file(two_pic)

    # face_recognition.face_locations(img,number_of_times_to_upsample=1,model='hog')
    # 返回图像中每张人脸的人脸特征位置列表；
    # number_of_times_to_upsample – 对图像进行多少次上采样以查找人脸。数字越大，人脸越小；
    # model – "hog"不太准确，但在CPU上更快。"cnn"是GPU / CUDA加速的一个更准确的深度学习模型。

    Log("face_locations - 人脸定位中" )
    face_1_locations = face_recognition.face_locations(face_1)

    if two_pic_locations:
        face_2_locations = two_pic_locations
    else:
        face_2_locations = face_recognition.face_locations(face_2)

    Log("face_encodings - 人脸数据编码中")
    # face_recognition.face_encodings(face_image, known_face_locations=None, num_jitters=1, model='small')
    # 返回图像中每张人脸的 128 维人脸编码。
    # known_face_locations - 可选 - 每个面孔的边界框（如果已经知道它们）- face_locations。
    # num_jitters – 计算编码时重新采样人脸的次数。越高越准确，但速度越慢（即 100 表示慢 100 倍）。
    # model – “large” (默认) 或 “small”仅返回5个点，但速度更快。

    face_1_encoding = face_recognition.face_encodings(face_1, known_face_locations=face_1_locations, num_jitters=1)[0]
    face_2_encodings = face_recognition.face_encodings(face_2, known_face_locations=face_2_locations, num_jitters=1)

    array_face_locations = []
    for  i in range(len(face_2_encodings)):
        current_encode =face_2_encodings[i]
        Log("compare_faces - 人脸数据匹配中")
        # tolerance 越低，匹配度数要求越高 
        # tolerance – 将人脸之间的距离视为匹配。越低越严格。0.6 是典型的最佳值。
        result = face_recognition.compare_faces([face_1_encoding],current_encode,tolerance=0.45)
        if result[0]:
            array_face_locations.append({"location":face_2_locations[i]})

    return array_face_locations

def Log(msg,result=None):
    global LOGO_SHOW

    if LOGO_SHOW:
        print(msg,result)

def showFace(pic_path=''):
    img = face_recognition.load_image_file(pic_path)
    face_landmarks_list = face_recognition.face_landmarks(img)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            Log(face_landmarks[facial_feature])
    
def faceDemo(picDir='',test_pic=''):
    pic_list= os.listdir(picDir)
    # 目标图像
    image = face_recognition.load_image_file(test_pic)

    # 目标图像
    test_pic_locations = face_recognition.face_locations(image)
    
    # 匹配文件计数器
    file_count = 0
    face_map={}
    # 对比所有人脸库
    for root, dirs, files in os.walk(picDir):
        for file in files:
            if file.endswith('.jpg'):
                file_count = file_count + 1
                one_pic_path = picDir + file
                locations = demoFunc(one_pic_path,test_pic,test_pic_locations)
                one_name = file.split(".")[1]
                if locations:
                    print("this Person is", one_name)
                    for face in locations:
                        face['persion_name'] = one_name

                        face_key = face['location']    

                        # 封装人脸定位与匹配的人脸名称 ：  { location:[ {location,persion_name} ] }  
                        if face_key in face_map:
                            face_map[face_key].append(face)
                        else:
                            face_map[face_key] = [face]    
                # else:
                #     print("this Person is not",one_name )

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
  
        print(list[0]['location'],persion_text)

    # 显示人脸库信息
    print("人脸库：", file_count)
    print("目标图像可识别人脸数", len(test_pic_locations))
    print("人脸比对次数", file_count * len(test_pic_locations)) 
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
    print("程序运行耗时：", round(run_time,2), "秒")

    
