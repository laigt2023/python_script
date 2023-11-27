from sanic.response import json, text
from sanic import Sanic, request
import cv2 as cv
from PIL import Image
from io import BytesIO
import io
import numpy as np
import codecs
import main as FACE_RECOGITION
import time
import os
import base64
import schedule
import datetime
import threading

PORT=8032
HOST="0.0.0.0"
app = Sanic("face-ai")
app.config.HEALTH = True

IS_SHOW_REPORT_IMG = False

# 最大记录日志数(/天）
LOG_RECORD_DAY = 30

# 定时任务运行 False-不运行 True运行
task_running=False

def testTask():
    print("testTask")

# 定义一个线程来运行定时任务
def run_schedule():
    global task_running
    while task_running:
        schedule.run_pending()
        time.sleep(1)

@app.route("/test", methods=["GET", "POST"])
async def test(request):
    status_code = 200
    res_dict = {"code": status_code,
                "message": "success"}
    
    time.sleep( 15 * 60 * 1)
   
    result = json(res_dict, status=status_code, ensure_ascii=False)
    return result

# 删除日志文件
def delete_old_logs():
    global LOG_RECORD_DAY
    clear_days = LOG_RECORD_DAY
    # if day:
    #    clear_days = day
    # 获取当前日期
    today = datetime.date.today()
    
    # 计算30天前的日期
    thirty_days_ago = today - datetime.timedelta(clear_days)

    current_path = os.path.dirname(os.path.realpath(__file__))
    dir=current_path + os.path.sep + "face_logs"

    del_message=""

    # 删除30天前的日志文件
    for file_name in os.listdir(dir):
        file_date = datetime.datetime.strptime(file_name, '%Y-%m-%d_face.log').date()
        if file_name.endswith(".log"):
            if file_date < thirty_days_ago:
                os.remove(os.path.join(dir, file_name))
                del_message+=f'Deleted {file_name}\n'
    if del_message:
        logMesg("清理"+ LOG_RECORD_DAY +"前旧日志文件")
        print(del_message)
        logMesg(del_message);            


@app.route("/face", methods=["GET", "POST"])
async def calculate_add(request):
    """ 分类 """
    if request.method == "POST":
        
        params = request.form if request.form else request.json
    elif request.method == "GET":
        params = request.args
    else:
        params = {}
    
    # print(params)
    if len(params['base64_code']) == 0:
        if 'file' not in request.files :
            result = json({'message': 'No image file'}, status=400)
            log_mesg = {'message':'No image file', "status":400,"url":request.url}
            # logMesg(request.url + " : " + json.dumps(result))
            logMesg(log_mesg)
            return result

        if len(request.files['file']) < 1:
            result = json({'message': 'No image file len < 1'}, status=400)
            log_mesg = {'message':'No image file len < 1', "status":400,"url":request.url}
            # logMesg(request.url + " : " + json.dumps(result))
            logMesg(log_mesg)
            return result
        
        file = request.files['file'][0]
        filename = file.name
        if filename == '':
            result = json({'message': 'filename is not exist'}, status=400)
            log_mesg = {'message':'filename is not exist', "status":400,"url":request.url}
            # logMesg(request.url + " : " + json.dumps(result))
            logMesg(log_mesg)
            return result

        # 读取文件内容到内存中
        image_bytes = file.body

        # 将字节流转换为OpenCV可读取的图片格式
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv.imdecode(img_array, cv.IMREAD_COLOR)
    else:
        filename = ''
        img = base64_to_image(params['base64_code'])

    # 展示图片
    # cv.imshow("insightface",img)
    # if cv.waitKey(0) & 0xFF ==("q"):
    #     cv.destroyAllWindows() 

    if 'sim' in params and params['sim']:
        comparison_value = float(params['sim'])
        # result,target_img = FACE_RECOGITION.funDb(img,comparison_value)
        result,target_img = FACE_RECOGITION.faceCpByDB(img,comparison_value)
        
    else:
        # result,target_img = FACE_RECOGITION.funDb(img)
        result,target_img = FACE_RECOGITION.faceCpByDB(img)
    
    respone_data = {
        "filename":filename,
        "face-info":result
    }
    
    status_code = 200
    res_dict = {"code": status_code,
                "data": respone_data,
                "message": "success"}
   
    result = json(res_dict, status=status_code, ensure_ascii=False)
    log_mesg = {"data": res_dict, "status":status_code,"url":request.url}
    # logMesg(request.url + " : " + json.dumps(result))
    logMesg(log_mesg)
    return result

@app.route("/face/db/reload", methods=["GET", "POST"])
async def reload_face_db(request):
    face_db = FACE_RECOGITION.reloadFaceDb()

    status_code = 200
    res_dict = {"code": status_code,
                "data": f"刷新重建人脸库，人脸库数据共: {len(face_db)} 个",
                "message": "success"}
   
    result = json(res_dict, status=status_code, ensure_ascii=False)
    log_mesg = {"data": res_dict, "status":status_code,"url":request.url}
    # logMesg(request.url + " : " + json.dumps(result))
    logMesg(log_mesg)
    return result

@app.route("/face/db/rebuild", methods=["GET", "POST"])
async def face_db_build(request):
    """ 分类 """
    if request.method == "POST":
        params = request.form if request.form else request.json
    elif request.method == "GET":
        params = {}

        
        for key in request.args.keys():
            print(request.args[key])
            if len(request.args[key]) > 1:
                params[key] = request.args[key]
            elif len(request.args[key]) == 1:
                params[key] = request.args[key][0]
    else:
        params = {}

    print(params)

    if params and "face_dir" in params:
        rebuild_face_db = FACE_RECOGITION.rebuildFaceDb(params["face_dir"])  
    else:
        rebuild_face_db = FACE_RECOGITION.rebuildFaceDb(None)

    status_code = 200
    res_dict = {"code": status_code,
                "data": f"重建人脸库，人脸库数据共: {len(rebuild_face_db)} 个",
                "message": "success"}
   
    result = json(res_dict, status=status_code, ensure_ascii=False)
    log_mesg = {"data": res_dict, "status":status_code,"url":request.url}
    # logMesg(request.url + " : " + json.dumps(result))
    logMesg(log_mesg)
    return result

@app.route("/face/log/clear", methods=["GET", "POST"])
async def face_log_clear(request):
    global LOG_RECORD_DAY
    old_day = LOG_RECORD_DAY
    """ 分类 """
    if request.method == "POST":
        
        params = request.form if request.form else request.json
    elif request.method == "GET":
        params = request.args
    else:
        params = {}

    if params and "day" in params:
        print(params)
        LOG_RECORD_DAY = params['day']
    delete_old_logs()    



    status_code = 200
    res_dict = {"code": status_code,
                "data": f"手动成功清除{LOG_RECORD_DAY}天前的日志文件",
                "message": "success"}
   
    result = json(res_dict, status=status_code, ensure_ascii=False)
    log_mesg = {"data": res_dict, "status":status_code,"url":request.url}
    # logMesg(request.url + " : " + json.dumps(result))
    logMesg(log_mesg)

    # 恢复默认配置
    LOG_RECORD_DAY=old_day
    return result

# base64转图片
def base64_to_image(encoded_string):
    # 写入日志文件  w-重新 a-追加
    with open("./web_post_base64.text", 'w', encoding='utf-8') as txt_file:
        txt_file.write(encoded_string)

    # decoded_bytes = base64.b64decode(encoded_string)
    # image = Image.open(BytesIO(decoded_bytes))
    img_b64decode = base64.b64decode(encoded_string)  # base64解码
 
    image = io.BytesIO(img_b64decode)
    img = Image.open(image)
    image = img
    return cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB)

# 显示处理后的人脸以及异常事件图片
@app.route("/face_report", methods=["GET", "POST"])
async def show_face(request):
    global IS_SHOW_REPORT_IMG
    """ 分类 """
    if request.method == "POST":
        
        params = request.form if request.form else request.json
    elif request.method == "GET":
        params = request.args
    else:
        params = {}
    # print(params)
    alarmTime = params['alarmTime']
    img = base64_to_image(params['alarmPicture'])
    info_array = params['info']
    # print(info_array)


    # 是否展示图片
    IS_SHOW_IMG = IS_SHOW_REPORT_IMG
    # 只展示识别到人脸的数据标记量
    is_show_face = False
    # 是否展示全部图片
    show_all = True

    if show_all:
        is_show_face = True
    for num in range(len(info_array)):
        current_info = info_array[num]
        print(current_info)
        if 'faceBox' in current_info:
            is_show_face = True
           
            face = current_info['faceBox']
            print("face",face)
            if face:
                face_position = face.split(",")
                cv.rectangle(img, (int(face_position[0]), int(face_position[1])), (int(face_position[2]), int(face_position[3])), (0, 255, 0), 2) 
                # 人脸ID
                cv.putText(img,current_info['cardId'], (int(face_position[0]), int(face_position[1]) - 5 ),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)

        if 'coordinate' in current_info:
            event = current_info['coordinate']
            if event:
                event_position = event.split(",")
                cv.rectangle(img, (int(event_position[0]), int(event_position[1])), (int(event_position[2]), int(event_position[3])), (0, 0, 255), 2)  

    if IS_SHOW_IMG:
        if is_show_face:

            cv.imshow('frame_' + str(alarmTime),img)

            if cv.waitKey(0) & 0xFF ==("q"):
                    cv.destroyAllWindows()

    status_code = 200
    res_dict = {"code": status_code,
                "data": params,
                "message": "success"}
   
    result = json(res_dict, status=status_code, ensure_ascii=False)
    # log_mesg = {"data": res_dict, "status":status_code,"url":request.url}
    # logMesg(request.url + " : " + json.dumps(result))
    # logMesg(log_mesg)
    return result            

# 统一日志输出
def logMesg(msg):
    # 获取当前路径
    today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir=current_path + os.path.sep + "face_logs"
    log_file_path = f"{dir + os.path.sep + today}_face.log"
    # 创建日志文件夹
    if(os.path.exists(dir) == False):
        os.mkdir(dir)

    # 写入日志文件  w-重新 a-追加
    with open(log_file_path, 'a', encoding='utf-8') as txt_file:
        result = f"{current_time} :{msg}"
        txt_file.write(f"{result}\n")
        print(result)

if __name__ == "__main__":
    # 开启定时任务
    task_running = True
  
    # schedule.every(5).seconds.do(testTask)  
    # 启动线程每天凌晨1点执行delete_old_logs函数
    schedule.every().day.at("01:00").do(delete_old_logs)  
    t = threading.Thread(target=run_schedule)
    t.start()

    app.run(single_process=True,
            access_log=True,
            host=HOST,
            port=PORT,
            workers=1,
            )
    # 关闭定时任务
    task_running = False
    