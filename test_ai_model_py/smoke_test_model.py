import sys 
sys.path.append("..")  
import ai_predict_test as AI_API 

# 对比2个模型识别率 -【吸烟检测】
# 目标模型输出到./target_xml
# 比较模型输出到./compare_xml
# COMPARE_AI_TASK_ID 为空时，只输出目标模型的xml，统计时无对比数据

# 无法识别中文路径，需要转换成英文路径 - 如开启jpeg图片输出的话，需要转换成英文路径 
# 开启方式   AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID,True) True-输出JPEG False-关闭
img_dir=r"D:\工作交接\智慧工地\吸烟视频\中医院\0802\smoke\somke_2_input"
# f41ee782-f34a-465a-a34b-7cddd50ea1f9 - 香烟图片识别V1-NEW（目标模型）
# 530cb7d0-8c0d-47ee-a2f3-a20e11bbad8e - 抽烟检测V2（New）
TARGET_AI_TASK_ID="530cb7d0-8c0d-47ee-a2f3-a20e11bbad8e"
COMPARE_AI_TASK_ID="f41ee782-f34a-465a-a34b-7cddd50ea1f9"
# 主函数
if __name__ == "__main__":
    # 是否跳过推理过程，直接输出上次推理结果
    AI_API.setSkipPredict(False)
    AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID)