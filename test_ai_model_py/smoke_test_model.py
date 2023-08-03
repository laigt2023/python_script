import sys 
sys.path.append("..")  
import ai_predict_test as AI_API 

# 对比2个模型识别率 -【吸烟检测】
# 目标模型输出到./target_xml
# 比较模型输出到./compare_xml
# COMPARE_AI_TASK_ID 为空时，只输出目标模型的xml，统计时无对比数据

img_dir=r"D:\工作交接\智慧工地\吸烟视频\中医院\0802\smoke\somke_0_input"
TARGET_AI_TASK_ID="e0b75053-52ec-4023-916f-a6d5ac523c64"
COMPARE_AI_TASK_ID=""
# 主函数
if __name__ == "__main__":
    # 是否跳过推理过程，直接输出上次推理结果
    AI_API.setSkipPredict(False)
    AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID)