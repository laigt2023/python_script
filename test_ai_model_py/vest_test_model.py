import sys 
sys.path.append("..")  
import ai_predict_test as AI_API 

img_dir=r"D:\工作交接\智慧工地\训练集\反光衣\SafetyVest\0727\GD_vest_train_20230728113945\opt\gongdi\biaozhu\image\val"
#img_dir=r"D:\工作交接\智慧工地\训练集\反光衣\SafetyVest\0731\GD_vest_train_20230731145504\opt\gongdi\biaozhu\image\val"
TARGET_AI_TASK_ID="e30bb34a-dcd9-4d95-8714-38df8d680815"
COMPARE_AI_TASK_ID="f07506b0-eb16-464c-8390-0639a6c053ca"
# 主函数
if __name__ == "__main__":
    AI_API.setSkipPredict(False)
    AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID)