import sys 
sys.path.append("..")  
import ai_predict_test as AI_API 

img_dir=r"D:\工作交接\智慧工地\吸烟视频\中医院\0802\smoke\somke_0_input"
TARGET_AI_TASK_ID="e0b75053-52ec-4023-916f-a6d5ac523c64"
COMPARE_AI_TASK_ID=""
# 主函数
if __name__ == "__main__":
    AI_API.setSkipPredict(False)
    AI_API.check_ai_models(img_dir,TARGET_AI_TASK_ID,COMPARE_AI_TASK_ID)