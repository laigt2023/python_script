import unittest
import cv2 as cv
import numpy as np
import time
import os
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# 替换成你选择的中文字体路径
font_path = r'd:\msyh.ttc'

# 设置中文字体
chinese_font = FontProperties(fname=font_path)

# 使用中文字体
plt.rcParams['font.family'] = chinese_font.get_name()

class IMAGE_ORB():
    def computeAndShow(queryImgPath,trainImgPath,isShowImg=False,roi={},subplot=None, rows =1 ,cols = 1):
        start_time = time.time()
        # 创建ORB检测器，设置检测器参数nfeatures=3000 - 特征点最大值 fastThreshold：FAST检测阈值，用于确定一个像素是否是关键点。默认值为20。
        # fastThreshold 通过比较中心像素点和周围一圈像素点的灰度值，快速判断是否为角点。
        # nlevels 较大的 nlevels 会导致更多的图像金字塔层数，允许在更广泛的尺度上进行特
        # 较小的 scaleFactor 会导致金字塔层数增加，从而使得在更多尺度上检测到关键点。这对于处理不同尺寸的特征物体或图像比例变化较大的场景非常有用。
        # edgeThreshold：边界阈值，用于决定图像边界处是否要舍弃特征点。调整这个值可能会影响是否检测到边缘附近的关键点。

        orb = cv.ORB_create(nfeatures=10000,fastThreshold=10,scaleFactor=2.5,nlevels=2,edgeThreshold=15,patchSize=15,WTA_K=4,scoreType=cv.ORB_FAST_SCORE)
        # orb = cv.ORB_create(nfeatures=1000,WTA_K=4,scoreType=cv.ORB_FAST_SCORE)
        # orb = cv.ORB_create(nfeatures=10000,fastThreshold=40,scaleFactor=1.2,nlevels=8,edgeThreshold=31,patchSize=31,WTA_K=2,scoreType=cv.ORB_FAST_SCORE)
        # orb = cv.ORB_create(nfeatures=10000,WTA_K=2)

        queryImg = cv.imread(queryImgPath, cv.IMREAD_GRAYSCALE)
        trainImg = cv.imread(trainImgPath, cv.IMREAD_GRAYSCALE)

        # 设置ROI的范围，这里假设你只想在图像的一部分区域中检测关键点
        # img[y1:y2,x1:yx2]
        if roi != None: 
            queryImgRoi = queryImg[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
            trainImgRoi = trainImg[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
        else:
            queryImgRoi = queryImg
            trainImgRoi = trainImg

        # cv.imshow("0", queryImgRoi)
        # cv.waitKey(0) 

        kp_query, des_query = orb.detectAndCompute(queryImgRoi, None)
        kp_train, des_train = orb.detectAndCompute(trainImgRoi, None)

        # 使用BFMatcher进行特征匹配 ratio 设置为 0.8 用于调整比例测试的阈值
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_query, des_train)


        # 使用knnMatch FMatcher 替换获取匹配
        # bf = cv.BFMatcher()
        # matches = bf.knnMatch(des_query, des_train, k=2)

        # # 比例测试的阈值
        # ratio_threshold = 1.1

        # # 应用比例测试
        # good_matches = []
        # for m, n in matches:
        #     if m.distance < ratio_threshold * n.distance:
        #         good_matches.append(m)

        # matches = good_matches

        # 根据匹配结果排序
        matches = sorted(matches, key=lambda x: x.distance)     

        # 计算相似度
        similarity = len(matches) / len(kp_query)
    
        print(f"ORB特征点数量: img1：{  len(kp_query) } img2：{ len(kp_train) }")
        print(f"特征点匹配数量: { len(matches) }")
        print(f"匹配率: { round(similarity * 100 ,2)  }%")
        print(f"耗时: {round(time.time() - start_time,4)} 秒")

        if isShowImg:
            # 获取不匹配的关键点 根据distance 在使用匹配器（比如暴力匹配器BFMatcher）进行特征点匹配时，匹配的程度通常由距离来衡量。距离越小，表示两个特征点越相似。
            matching_query_kp = []
            matching_train_kp = []
            new_matches = []
            # match.distance 表示两个特征描述子之间的距离，通常情况下，距离越小表示两个特征越相似
            for match in matches:
                if match.distance < 1000:
                    matching_query_kp.append(kp_query[match.queryIdx])
                    matching_train_kp.append(kp_train[match.trainIdx])
                    new_matches.append(match)
         
            # 获取匹配的关键点下标对照
            query_unique_keys_set = {obj.queryIdx for obj in new_matches}
            train_unique_keys_set = {obj.trainIdx for obj in new_matches}
            print(f"特征点匹配数：{len(query_unique_keys_set)}")

            
            # 原图-训练图：绘制匹配的关键点
            non_matching_train = get_non_duplicate_objects(kp_train,train_unique_keys_set,True)
            train_marked = cv.drawKeypoints(trainImgRoi, non_matching_train, None, color=(0, 255, 0), flags=2)

            # 原图-训练图：的不匹配的关键点
            non_not_matching_train = get_non_duplicate_objects(kp_train,train_unique_keys_set,False)
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

            # 将原图中的黄点画到 推理图上 -add
            query_all_marked = cv.drawKeypoints(query_all_marked, non_not_matching_train, None, color=(255, 255, 0), flags=2)

            if rows != None and cols != None and rows > 0 and cols > 0:
                # 推理图 不匹配的红点
                calculate_keypoint_density(query_all_marked,non_not_matching_query, rows, cols,type="diff")
                # 推理图 匹配的绿点 -add
                calculate_keypoint_density(query_all_marked,matching_query, rows, cols,type="same")

                # 原图 不匹配的黄点 -add
                calculate_keypoint_density(train_marked,non_not_matching_train, rows, cols,type=None)
                # 原图 匹配的绿点 -add
                calculate_keypoint_density(train_marked,matching_train_kp, rows, cols,type="same")

            # 显示结果
            if roi != None:
                plt.suptitle(f'ORB算法 ROI:  ({roi["x1"]},{roi["y1"]}) : ({roi["x2"]},{roi["y2"]})  { roi["x2"] - roi["x1"] }  X  { roi["y2"] - roi["y1"] }' , fontsize=16)

            if subplot == None:
                train_marked_title = f'原图 {os.path.basename(trainImgPath)}（关键点:{len(kp_train)}）  \n黄色：{len(non_not_matching_train)} 关键点（不与推理图匹配） \n绿色：{len(matching_query_kp)} （与推理图匹配的关键点）'
                plt.subplot(121), 
                plt.imshow(train_marked),
                plt.title(train_marked_title)
                
                
                query_title = f'推理图 {os.path.basename(queryImgPath)}（关键点:{len(kp_query)}）  \n绿色: { len(matching_query)}（与原图匹配） 匹配率：{round(len(matching_query)/len(kp_query) * 100 ,2)}%   红色：{ len(non_not_matching_query)} （与原图不匹配）差异率：{round(len(non_not_matching_query)/len(kp_query) * 100 ,2)}%'
                plt.subplot(122), 
                plt.imshow(query_all_marked), 
                plt.title(query_title)
            else:    
                train_marked_title = f'原图 {os.path.basename(trainImgPath)}（关键点:{len(kp_train)}）  \n黄色：{len(non_not_matching_train)} 关键点（不与推理图匹配） \n绿色：{len(matching_query_kp)} （与推理图匹配的关键点）'
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

    def compute(self,queryImgPath,trainImgPath,roi=None):
        print("compute",queryImgPath,trainImgPath,roi)
        # 创建ORB检测器，设置检测器参数nfeatures=3000 - 特征点最大值 fastThreshold：FAST检测阈值，用于确定一个像素是否是关键点。默认值为20。
        # fastThreshold 通过比较中心像素点和周围一圈像素点的灰度值，快速判断是否为角点。
        # nlevels 较大的 nlevels 会导致更多的图像金字塔层数，允许在更广泛的尺度上进行特
        # 较小的 scaleFactor 会导致金字塔层数增加，从而使得在更多尺度上检测到关键点。这对于处理不同尺寸的特征物体或图像比例变化较大的场景非常有用。
        # edgeThreshold：边界阈值，用于决定图像边界处是否要舍弃特征点。调整这个值可能会影响是否检测到边缘附近的关键点。

        orb = cv.ORB_create(nfeatures=10000,fastThreshold=10,scaleFactor=2.5,nlevels=2,edgeThreshold=15,patchSize=15,WTA_K=4,scoreType=cv.ORB_FAST_SCORE)
        # orb = cv.ORB_create(nfeatures=10000,fastThreshold=40,scaleFactor=1.2,nlevels=8,edgeThreshold=31,patchSize=31,WTA_K=2,scoreType=cv.ORB_FAST_SCORE)

        queryImg = cv.imread(queryImgPath, cv.IMREAD_GRAYSCALE)
        trainImg = cv.imread(trainImgPath, cv.IMREAD_GRAYSCALE)

        # 设置ROI的范围，这里假设你只想在图像的一部分区域中检测关键点
        # img[y1:y2,x1:yx2]
        if roi != None: 
            queryImgRoi = queryImg[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
            trainImgRoi = trainImg[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
        else:
            queryImgRoi = queryImg
            trainImgRoi = trainImg

        # cv.imshow("0", queryImgRoi)
        # cv.waitKey(0) 

        kp_query, des_query = orb.detectAndCompute(queryImgRoi, None)
        kp_train, des_train = orb.detectAndCompute(trainImgRoi, None)

        # 使用BFMatcher进行特征匹配 ratio 设置为 0.8 用于调整比例测试的阈值
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_query, des_train)

        # 根据匹配结果排序
        matches = sorted(matches, key=lambda x: x.distance)     


        # 获取不匹配的关键点 根据distance 在使用匹配器（比如暴力匹配器BFMatcher）进行特征点匹配时，匹配的程度通常由距离来衡量。距离越小，表示两个特征点越相似。
        matching_query_kp = []
        matching_train_kp = []
        new_matches = []
        # match.distance 表示两个特征描述子之间的距离，通常情况下，距离越小表示两个特征越相似
        for match in matches:
            if match.distance < 1000:
                matching_query_kp.append(kp_query[match.queryIdx])
                matching_train_kp.append(kp_train[match.trainIdx])
                new_matches.append(match)
        
        # 获取匹配的关键点下标对照
        query_unique_keys_set = {obj.queryIdx for obj in new_matches}
        train_unique_keys_set = {obj.trainIdx for obj in new_matches}

        # train
        non_matching_train = get_non_duplicate_objects(kp_train,train_unique_keys_set,True)
        non_not_matching_train = get_non_duplicate_objects(kp_train,train_unique_keys_set,False)
        # query
        matching_query= get_non_duplicate_objects(kp_query,query_unique_keys_set,True)
        non_not_matching_query = get_non_duplicate_objects(kp_query,query_unique_keys_set,False)
        
        kp_query_num = len(kp_query)
        kp_train_num = len(kp_train)
        query_same_num = len(matching_query)
        train_same_num = len(non_matching_train)
        query_diff_num = len(non_not_matching_query)
        train_diff_num = len(non_not_matching_train)

        return {
            "queryKpNum":kp_query_num,
            "querySameKpNum":query_same_num,
            "queryDiffKpNum":kp_query_num - query_diff_num,
            "querySameRate": round( query_same_num / kp_query_num, 4 ),
            "queryDiffRate": round( query_diff_num / kp_query_num, 4 ),
            "trainKpNum":kp_train_num,
            "trainSameKpNum":train_same_num,
            "trainDiffKpNum":kp_train_num - train_diff_num,
            "trainSameRate": round( train_same_num / kp_train_num, 4 ),
            "trainDiffRate": round( train_diff_num / kp_train_num, 4 ),
        }

# 划分为 rows * cols 个区域,每个区域坐标符合kp数量，type类型：  same-绿色/diff-红色/None-白色  
def calculate_keypoint_density(color_img,kp, rows, cols,type="same"):
    height, width = color_img.shape[:2]
    color = (255, 255, 0)
    x_offset = 0
    if type == 'same':
        color = (0, 255, 0)
        x_offset = 40

    if type == 'diff':
        color = (255, 0, 0)

    # 计算每个区域的宽度和高度
    region_width =  width // cols
    region_height = height // rows

    # 初始化密度矩阵
    diff_matrix = np.zeros((rows, cols))
    # 遍历每个匹配点
    for k in kp:
        # 确定匹配点属于哪个区域
        col_index = int(k.pt[0] // region_width)
        row_index = int(k.pt[1] // region_height)
        # 增加对应区域的密度
        diff_matrix[row_index, col_index] += 1
    
    # 在图像上显示每个区域的关键点匹配数量
    for i in range(rows):
        for j in range(cols):
            x, y = int(j * (color_img.shape[1] / cols)), int(i * (color_img.shape[0] / rows))
            text = f"{int(diff_matrix[i, j])}"
            if int(text) != 0:
                cv.putText(color_img, text, (x + x_offset, y + 35), cv.FONT_HERSHEY_SIMPLEX, 1.8, color, 1, cv.LINE_AA)


    height, width = color_img.shape[:2]

    # 计算每个区域的宽度和高度
    region_width =  width // cols
    region_height = height // rows

    # 定义白色 (BGR格式)
    white_color = (255, 255, 255)

    # 在每个列画垂直线
    for i in range(1, cols):
        x = i * region_width
        cv.line(color_img, (x, 0), (x, height), white_color, 1)

    # 在每个行画水平线
    for j in range(1, rows):
        y = j * region_height
        cv.line(color_img, (0, y), (width, y), white_color, 1)
    

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
        ORB = IMAGE_ORB()
        #    img1Path = r'.\img\camera\img-6-x2.jpg'
        #    img2Path = r'.\img\camera\img-6-x1.jpg'

        # img[y1:y2,x1:yx2] 取左上角（750px，520px）到右下角（1500px,780px）区域
       

        test_11 = {
            "trainImgPath" : r'.\img\camera\img-19.jpg',
            "queryImgPath1" : r'.\img\camera\img-19-1.jpg',
            "queryImgPath2" : r'.\img\camera\img-19-2.jpg',
            "queryImgPath3" : r'.\img\camera\img-19-3.jpg',
            "queryImgPath4" : r'.\img\camera\img-19-x1.jpg',
            "queryImgPath5" : r'.\img\camera\img-19-x1-1.jpg',
            "roi" : {
                "x1":300,
                "y1":2100,
                "x2":2500,
                "y2":3000,
            },
            "roi" : {
                "x1":300,
                "y1":200,
                "x2":4300,
                "y2":2800,
            }
        }

        plt.figure(figsize=(16, 16))
        # 调整子图之间的垂直间距
        plt.subplots_adjust(hspace=0.5)
        IS_SHOW_IMG = False

        result = IMAGE_ORB.computeAndShow(test_11["queryImgPath1"], test_11["trainImgPath"],isShowImg = IS_SHOW_IMG,roi=test_11["roi"],subplot=321, rows = 10 ,cols = 20)
        result = IMAGE_ORB.computeAndShow(test_11["queryImgPath4"], test_11["trainImgPath"],isShowImg = IS_SHOW_IMG,roi=test_11["roi"],subplot=323, rows = 10 ,cols = 20)
        result = IMAGE_ORB.computeAndShow(test_11["queryImgPath5"], test_11["trainImgPath"],isShowImg = IS_SHOW_IMG,roi=test_11["roi"],subplot=325, rows = 10 ,cols = 20)

        if IS_SHOW_IMG:
            plt.show()
        self.assertEqual(result, None)
        
# if __name__ == '__main__':
#    unittest.main()        
