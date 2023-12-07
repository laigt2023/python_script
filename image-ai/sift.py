import unittest
import cv2 as cv
import numpy as np
import time
import os
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import SubplotTool
import json

# 替换成你选择的中文字体路径
font_path = r'd:\msyh.ttc'

# 设置中文字体
chinese_font = FontProperties(fname=font_path)

# 使用中文字体
plt.rcParams['font.family'] = chinese_font.get_name()

class ImageSURF():
    def SITF(img1Path,img2Path):
        start_time = time.time()
        img = cv.imread(img1Path)

        # 创建SIFT检测器
        sift = cv.SIFT_create()

        # 检测并计算图像中的特征点
        kp = sift.detect(img,None)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 创建一个名为img2的图像，用于显示检测到的特征点
        outImg = cv.drawKeypoints(gray, kp, img)  # 画出关键点

        # img2 = cv.imread(img2Path)
        # kp_train, des2 = sift.detectAndCompute(img2,None)

        print(f"耗时: {round(time.time() - start_time,4)} 秒")
        cv.imshow('sift', outImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def SITF_IMG(queryImgPath,trainImgPath,isShowImg=False,roi=None,subplot=None):
        start_time = time.time()
        
        queryImg = cv.imread(queryImgPath)
        trainImg = cv.imread(trainImgPath)
        
        # 将图片转换为灰度图像
        queryImgGray = cv.cvtColor(queryImg, cv.COLOR_BGR2GRAY)
        trainImgGray = cv.cvtColor(trainImg, cv.COLOR_BGR2GRAY)

        
        # 设置ROI的范围，这里假设你只想在图像的一部分区域中检测关键点
        # img[y1:y2,x1:yx2]
        if roi != None: 
            queryImgRoi = queryImgGray[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
            trainImgRoi = trainImgGray[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
        else:
            queryImgRoi = queryImgGray
            trainImgRoi = trainImgGray

        # 创建SIFT检测器
        # 创建SIFT检测器，设置检测器参数nfeatures=3000 - 特征点最大值 contrastThreshold=0.03 - 对比度阈值
        sift = cv.SIFT_create(nfeatures=4000,edgeThreshold=10, contrastThreshold=0.04)
        # 设置随机数种子
        np.random.seed(42)

        kp_query, des1 = sift.detectAndCompute(queryImgRoi, None)
        kp_train, des2 = sift.detectAndCompute(trainImgRoi, None)
          
        # k对最佳匹配
        # match.queryIdx = 0  # 查询图像中的关键点索引
        # match.trainIdx = 1  # 训练图像中的关键点索引
        # match.distance = 5  # 两个关键点之间的距离
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

       
    
        # 获取不匹配的关键点 根据distance 在使用匹配器（比如暴力匹配器BFMatcher）进行特征点匹配时，匹配的程度通常由距离来衡量。距离越小，表示两个特征点越相似。
        matching_query_kp = []
        matching_train_kp = []
        new_matches = []
        # match.distance 表示两个特征描述子之间的距离，通常情况下，距离越小表示两个特征越相似
        matching_query_inex_list = []
        matching_train_inex_list = []
        
        # 比率测试，舍弃不佳的匹配
        good_matches = []

        # best_match = match_pair[0] 获取最近邻匹配
        # second_best_match = match_pair[1] 获取次近邻匹配
        for best_match, second_best_match in matches:
            if best_match.distance < 0.7 * second_best_match.distance:
                good_matches.append(best_match)

        matches = good_matches
        print("good_matches",len(good_matches))
        # for match_pair in matches:
        #     for match in matches:
        #         if match.distance < 100:
        #             # if match.queryIdx not in matching_query_inex_list:
        #             #     matching_query_inex_list.append(match.queryIdx)
        #             #     matching_query_kp.append(kp_query[match.queryIdx])

        #             # if match.trainIdx not in matching_train_inex_list:    
        #             #     matching_train_inex_list.append(match.trainIdx)
        #             #     matching_train_kp.append(kp_train[match.trainIdx])

        #             # if match.queryIdx not in matching_query_inex_list and match.trainIdx not in matching_train_inex_list:
        #             #     new_matches.append(match)
                        
        #             matching_query_kp.append(kp_query[match.queryIdx])
        #             matching_train_kp.append(kp_train[match.trainIdx])   
        #             new_matches.append(match)

        
        for match in matches:
            if match.distance < 10000:
                matching_query_kp.append(kp_query[match.queryIdx])
                matching_train_kp.append(kp_train[match.trainIdx])   
                new_matches.append(match)

        # 获取匹配的关键点下标对照
        query_unique_keys_set = {obj.queryIdx for obj in new_matches}
        train_unique_keys_set = {obj.trainIdx for obj in new_matches}

        # 计算相似度
        similarity = len(query_unique_keys_set) / len(kp_query)
        print(f"SIFT特征点数量: img1：{  len(kp_query) } img2：{ len(kp_train) }")
        print(f"特征点匹配数量: { len(query_unique_keys_set) }")
        print(f"匹配率: { round(similarity * 100 ,2)  }%")
        print(f"耗时: {round(time.time() - start_time,4)} 秒")            

        isShowImg = True
        if isShowImg:
        
            # 原图-训练图：绘制匹配的关键点 绿色-匹配的点
            train_marked = cv.drawKeypoints(trainImgRoi, matching_train_kp, None, color=(0, 255, 0), flags=2)

            # 原图-训练图：的不匹配的关键点 黄色-匹配的点
            non_not_matching_train = get_non_duplicate_objects(kp_train,train_unique_keys_set,False)
            non_matching_train = get_non_duplicate_objects(kp_train,train_unique_keys_set,True)
            train_marked = cv.drawKeypoints(train_marked, non_not_matching_train, None, color=(255, 255, 0), flags=2)


            # 推理图：全量的匹配的关键点 绿色-匹配的点
            
            matching_query= get_non_duplicate_objects(kp_query,query_unique_keys_set,True)
            query_marked = cv.drawKeypoints(queryImgRoi, matching_query, None, color=(0, 255, 0), flags=2)

            # 推理图：全量的不匹配的关键点  红色- 不匹配的点
            non_not_matching_query = get_non_duplicate_objects(kp_query,query_unique_keys_set,False)
            query_not_marked = cv.drawKeypoints(queryImgRoi, non_not_matching_query, None, color=(255, 0, 0), flags=0)



            # 推理-全量图
            query_all_marked = cv.drawKeypoints(queryImgRoi, matching_query, None, color=(0, 255, 0), flags=0)
            # 推理-全量图-加入不匹配的点
            query_all_marked = cv.drawKeypoints(query_all_marked, non_not_matching_query, None, color=(255, 0, 0), flags=0)

    
            # 显示结果
            if roi != None:
                plt.suptitle(f'SIFT算法 ROI:  ({roi["x1"]},{roi["y1"]}) : ({roi["x2"]},{roi["y2"]})  { roi["x2"] - roi["x1"] }  X  { roi["y2"] - roi["y1"] }' , fontsize=16)

            if subplot == None:
                train_marked_title = f'原图 {os.path.basename(trainImgPath)}（关键点:{len(kp_train)}）  \n黄色：{len(non_not_matching_train)} 关键点（不与推理图匹配） \n绿色：{len(non_matching_train)} （与推理图匹配的关键点）'
                plt.subplot(121), 
                plt.imshow(train_marked),
                plt.title(train_marked_title)
                
                
                query_title = f'推理图 {os.path.basename(queryImgPath)}（关键点:{len(kp_query)}）  \n绿色: { len(matching_query)}（与原图匹配） 匹配率：{round(len(matching_query)/len(kp_query) * 100 ,2)}%   红色：{ len(non_not_matching_query)} （与原图不匹配）差异率：{round(len(non_not_matching_query)/len(kp_query) * 100 ,2)}%'
                plt.subplot(122), 
                plt.imshow(query_all_marked), 
                plt.title(query_title)
            else:    
                train_marked_title = f'原图 {os.path.basename(trainImgPath)}（关键点:{len(kp_train)}）  \n黄色：{len(non_not_matching_train)} 关键点（不与推理图匹配） \n绿色：{len(non_matching_train)} （与推理图匹配的关键点）'
                plt.subplot(subplot), 
                plt.imshow(train_marked),
                plt.title(train_marked_title)
                
                
                query_title = f'推理图 {os.path.basename(queryImgPath)}（关键点:{len(kp_query)}）  \n绿色: { len(matching_query)}（与原图匹配） 匹配率：{round(len(matching_query)/len(kp_query) * 100 ,2)}%   \n红色：{ len(non_not_matching_query)} （与原图不匹配）差异率：{round(len(non_not_matching_query)/len(kp_query) * 100 ,2)}%'
                plt.subplot(subplot + 1), 
                plt.imshow(query_all_marked), 
                plt.title(query_title)

                
                # plt.subplot(223), 
                # plt.imshow(query_marked), 
                # plt.title(f'推理图 {os.path.basename(trainImgPath)}（关键点:{len(kp_query)}）   \n绿色：{ len(matching_query)} （与原图匹配的关键点） 匹配率：{round(len(matching_query)/len(kp_query) * 100 ,2)}%' )
                
                # plt.subplot(224), 
                # plt.imshow(query_not_marked), 
                # plt.title(f'推理图 {os.path.basename(trainImgPath)}（关键点:{len(kp_query)}）   \n红色：{ len(non_not_matching_query)} （与原图不匹配的关键点）差异率：{round(len(non_not_matching_query)/len(kp_query) * 100 ,2)}%')
            return {
                "train_marked":train_marked,
                "train_marked_title":train_marked_title,
                "query_marked":query_all_marked,
                "query_marked_title":train_marked_title,
            }
       
# 获取去重后的数组 is_include 是否包含，True获取包含在【unique_keys_set】的数组， False获取不包含在【unique_keys_set】数组
def get_non_duplicate_objects(array, unique_keys_set, is_include=False):
    # 从 array 中筛选出不在 unique_keys_set 中存在的对象
    result = []
  
    for index, value in enumerate(array):
        if is_include:
            if index in unique_keys_set:
                result.append(value)
        else:
            if index not in unique_keys_set:
                result.append(value)        
    return result

class TestImageSURF(unittest.TestCase):

   def test_SURF(self):
        img1Path = r'.\img\camera\img-7-x1.jpg'
        img2Path = r'.\img\camera\img-7.jpg'

        test_1 = {
                "queryImg1Path" : r'.\img\camera\img-8-1.jpg',
                "trainImgPath" : r'.\img\camera\img-8.jpg',
                "roi" : {
                    "x1":750,
                    "y1":520,
                    "x2":1500,
                    "y2":780,
                }
            }

        test_2 = {
            "queryImg1Path" : r'.\img\camera\img-10-x1.jpg',
            "trainImgPath" : r'.\img\camera\img-10-x2.jpg',
            "roi" : {
                "x1":200,
                "y1":300,
                "x2":1500,
                "y2":800,
            }
        }

        test_3 = {
            "queryImg1Path" : r'.\img\camera\img-m2-x1.jpg',
            "trainImgPath" : r'.\img\camera\img-m2.jpg',
            "roi" : {
                "x1":250,
                "y1":300,
                "x2":1000,
                "y2":1300,
            }
        }

        test_4 = {
            "queryImg1Path" : r'.\img\camera\img-11-1.jpg',
            "trainImgPath" : r'.\img\camera\img-11.jpg',
            "roi" : {
                "x1":550,
                "y1":300,
                "x2":1500,
                "y2":730,
            }
        }

        test_5 = {
            "queryImg1Path" : r'.\img\camera\img-12-x2.jpg',
            "trainImgPath" : r'.\img\camera\img-12.jpg',
            "roi" : {
                "x1":550,
                "y1":300,
                "x2":2000,
                "y2":1000,
            }
        }

        test_6 = {
            "queryImgPath" : r'.\img\camera\img-14.jpg',
            "trainImgPath" : r'.\img\camera\img-13.jpg',
            "roi" : {
                "x1":500,
                "y1":2000,
                "x2":3000,
                "y2":3000,
            }
        }

        test_7 = {
            "trainImgPath" : r'.\img\camera\img-14.jpg',
            "queryImgPath" : r'.\img\camera\img-14-x1.jpg',
            "queryImgPath2" : r'.\img\camera\img-14-x1-1.jpg',
            "queryImgPath3" : r'.\img\camera\img-14-x2.jpg',
            "roi" : {
                "x1":200,
                "y1":1800,
                "x2":3000,
                "y2":3300,
            }
        }

        test_8 = {
            "trainImgPath" : r'.\img\camera\img-15.jpg',
            "queryImgPath1" : r'.\img\camera\img-15-1.jpg',
            "queryImgPath2" : r'.\img\camera\img-15-x1.jpg',
            "queryImgPath3" : r'.\img\camera\img-15-x2.jpg',
            "queryImgPath4" : r'.\img\camera\img-15-x3.jpg',
            "roi" : {
                "x1":10,
                "y1":1500,
                "x2":3000,
                "y2":3700,
            }
        }

        test_9 = {
            "trainImgPath" : r'.\img\camera\img-17.jpg',
            "queryImgPath1" : r'.\img\camera\img-17-1.jpg',
            "queryImgPath2" : r'.\img\camera\img-17-x1.jpg',
            "queryImgPath3" : r'',
            "queryImgPath4" : r'',
            "roi" : {
                "x1":100,
                "y1":900,
                "x2":2000,
                "y2":1300,
            }
        }

        test_10 = {
            "trainImgPath" : r'.\img\camera\img-18.jpg',
            "queryImgPath1" : r'.\img\camera\img-18-1.jpg',
            "queryImgPath2" : r'.\img\camera\img-18-2.jpg',
            "queryImgPath3" : r'.\img\camera\img-18-x1.jpg',
            "queryImgPath4" : r'.\img\camera\img-18-x1-1.jpg',
            "roi" : {
                "x1":300,
                "y1":2100,
                "x2":2500,
                "y2":3000,
            }
        }
        test = test_10
 

        plt.figure(figsize=(16, 16))
        # 调整子图之间的垂直间距
        plt.subplots_adjust(hspace=0.5)
        # result = ImageSURF.SITF_IMG(test["queryImgPath1"], test["trainImgPath"],isShowImg = True,roi=test["roi"],subplot=241)
        # result = ImageSURF.SITF_IMG(test["queryImgPath2"], test["trainImgPath"],isShowImg = True,roi=test["roi"],subplot=243)
        # result = ImageSURF.SITF_IMG(test["queryImgPath3"], test["trainImgPath"],isShowImg = True,roi=test["roi"],subplot=245)
        # result = ImageSURF.SITF_IMG(test["queryImgPath4"], test["trainImgPath"],isShowImg = True,roi=test["roi"],subplot=247)
   
        result = ImageSURF.SITF_IMG(test_9["queryImgPath1"], test_9["trainImgPath"],isShowImg = True,roi=test_9["roi"],subplot=221)
        result = ImageSURF.SITF_IMG(test_9["queryImgPath2"], test_9["trainImgPath"],isShowImg = True,roi=test_9["roi"],subplot=223)
        plt.show()
        self.assertEqual(result, None)

if __name__ == '__main__':
   unittest.main()