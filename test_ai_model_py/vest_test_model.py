import sys 
sys.path.append("..")  
import ai_predict_test as AI_API 

# 对比2个模型识别率 -【反光衣】
# 目标模型输出到./target_xml
# 比较模型输出到./compare_xml
# COMPARE_AI_TASK_ID 为空时，只输出目标模型的xml，统计时无对比数据

# 无法识别中文路径，需要转换成英文路径 - 如开启jpeg图片输出的话，需要转换成英文路径 
# 开启方式   AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID,True) True-输出JPEG False-关闭
img_dir=r"D:\normal_clothes"  
#img_dir=r"D:\工作交接\智慧工地\训练集\反光衣\SafetyVest\0731\GD_vest_train_20230731145504\opt\gongdi\biaozhu\image\val"
TARGET_AI_TASK_ID="e30bb34a-dcd9-4d95-8714-38df8d680815"
COMPARE_AI_TASK_ID=""
# 主函数
if __name__ == "__main__":
    # 是否跳过推理过程，直接输出上次推理结果
    AI_API.setSkipPredict(False)
    AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID,True)