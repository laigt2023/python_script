import sys 
sys.path.append("..")  
import ai_predict_test as AI_API 

# 对比2个模型识别率 -【蓝色安全员】
# 目标模型输出到./target_xml
# 比较模型输出到./compare_xml
# COMPARE_AI_TASK_ID 为空时，只输出目标模型的xml，统计时无对比数据

# 无法识别中文路径，需要转换成英文路径 - 如开启jpeg图片输出的话，需要转换成英文路径 
# 开启方式   AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID,True) True-输出JPEG False-关闭
img_dir=r"D:\工作交接\智慧工地\误报数据\中医医院\TD_blue_helmet_train_20230726163353\opt\gongdi\biaozhu\image\val"
TARGET_AI_TASK_ID="8f97a571-22e5-41e3-afed-8f525d8338af"
COMPARE_AI_TASK_ID="e9e35528-71c7-4b62-9b88-8970bee55ab6"
# 主函数
if __name__ == "__main__":
    # 是否跳过推理过程，直接输出上次推理结果
    AI_API.setSkipPredict(False)
    AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID)