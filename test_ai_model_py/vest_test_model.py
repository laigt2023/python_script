import sys 
sys.path.append("..")  
import ai_predict_test as AI_API 

# 对比2个模型识别率 -【反光衣】
# 目标模型输出到./target_xml
# 比较模型输出到./compare_xml
# COMPARE_AI_TASK_ID 为空时，只输出目标模型的xml，统计时无对比数据

img_dir=r"E:\GD_vest_train_20230728113945\opt\gongdi\biaozhu\image\val"
#img_dir=r"D:\工作交接\智慧工地\训练集\反光衣\SafetyVest\0731\GD_vest_train_20230731145504\opt\gongdi\biaozhu\image\val"
TARGET_AI_TASK_ID="e30bb34a-dcd9-4d95-8714-38df8d680815"
COMPARE_AI_TASK_ID="f07506b0-eb16-464c-8390-0639a6c053ca"
# 主函数
if __name__ == "__main__":
    # 是否跳过推理过程，直接输出上次推理结果
    AI_API.setSkipPredict(False)
    AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID,True)