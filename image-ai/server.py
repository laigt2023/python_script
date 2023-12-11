from sanic.response import json, text
from sanic import Sanic
import numpy as np
import orb as ORB
import time
import os
import schedule
import datetime
import threading
from pydantic import BaseModel

PORT = 8033
HOST = "0.0.0.0"
app = Sanic("image-ai")
app.config.HEALTH = True

IS_SHOW_REPORT_IMG = False

# 最大记录日志数(/天）
LOG_RECORD_DAY = 30
LOG_FILE_DIR = 'logs'

# 定时任务运行 False-不运行 True运行
TASK_RUNNING=False

class ORBRequestModel(BaseModel):
    targetImg: str
    baseImg: str
    roi: object

@app.route("/orb/compute", methods=["GET", "POST"])
async def orb_compute(request):
    orb = ORB.IMAGE_ORB()
    start_time = time.time()
    try:
        # 从 request.json 中获取 POST 参数，确保它是一个 JSON 对象
        request_data = ORBRequestModel(**request.json)
        roi = None
        # 访问参数
        target_img = request_data.targetImg
        base_img = request_data.baseImg
        if request_data.roi:
            roi = request_data.roi

        if not os.path.exists(target_img):
            res_dict = {"code": 500,
                    "message": f"The file {target_img} does not exist."}
            logMesg(res_dict)
            return  res_dict
        
        if not os.path.exists(base_img):
            res_dict = {"code": 500,
                    "message": f"The file {base_img} does not exist."}
            logMesg(res_dict)
            return  res_dict

        computed_data = orb.compute(target_img,base_img,roi)
        

        status_code = 200
        res_dict = {"code": status_code,
                    "data": computed_data,
                    "message": "success"}

        result = json(res_dict, status=status_code, ensure_ascii=False)
        log_mesg = {"data": res_dict, "status":status_code,"url":request.url}
        logMesg(log_mesg)
        logMesg(f"开始时间：{start_time} 总耗时: {round(time.time() - start_time,4)} 秒")
        return result
    except Exception as e:
        # 处理参数解析错误等异常
        return json({"error": str(e)})

# 检测并创建日志目录
def create_log_dir():
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir=current_path + os.path.sep + LOG_FILE_DIR
    # 判断目录是否存在
    if not os.path.exists(dir):
        # 如果不存在，创建目录
        os.makedirs(dir)
        print(f"日志目录 {dir} 创建成功")

# 统一日志输出
def logMesg(msg):
    # 获取当前路径
    today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir=current_path + os.path.sep + LOG_FILE_DIR
    log_file_path = f"{dir + os.path.sep + today}_.log"
    # 创建日志文件夹
    if(os.path.exists(dir) == False):
        os.mkdir(dir)

    # 写入日志文件  w-重新 a-追加
    with open(log_file_path, 'a', encoding='utf-8') as txt_file:
        result = f"{current_time} :{msg}"
        txt_file.write(f"{result}\n")
        print(result)

# 定义一个线程来运行定时任务
def run_schedule():
    global TASK_RUNNING
    while TASK_RUNNING:
        schedule.run_pending()
        time.sleep(1)

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
    dir=current_path + os.path.sep + LOG_FILE_DIR

    del_message=""

    # 删除30天前的日志文件
    for file_name in os.listdir(dir):
        file_date = datetime.datetime.strptime(file_name, '%Y-%m-%d_.log').date()
        if file_name.endswith(".log"):
            if file_date < thirty_days_ago:
                os.remove(os.path.join(dir, file_name))
                del_message+=f'Deleted {file_name}\n'
    if del_message:
        logMesg("清理"+ LOG_RECORD_DAY +"前旧日志文件")
        print(del_message)
        logMesg(del_message);      

if __name__ == "__main__":
    # 开启定时任务
    TASK_RUNNING = True

    create_log_dir()

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
    TASK_RUNNING = False
    