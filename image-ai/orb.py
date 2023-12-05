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
    def showKeypoints(image_path):
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        # 初始化ORB检测器
        orb = cv.ORB_create(nfeatures=10000,fastThreshold=20)

        # 使用ORB找到关键点和描述符
        keypoints, descriptors = orb.detectAndCompute(image, None)

        # 在图像上绘制关键点
        image_with_keypoints = cv.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

        # 显示结果
        plt.imshow(image_with_keypoints, cmap='gray')
        plt.title('Image with ORB Keypoints')
        plt.axis('off')
        plt.show()
    def keypoints_match(img1Path,img2Path):
        img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE)


        orb = cv.ORB_create(nfeatures=10000,fastThreshold=20)
        
        # 设置相同的随机种子
        np.random.seed(42)
        kp_query, des1 = orb.detectAndCompute(img1, None)
        # 设置相同的随机种子
        np.random.seed(42)
        kp_train, des2 = orb.detectAndCompute(img2, None)

        cp_count = 0
        for index, kp in enumerate(kp_query):
            ckp = kp_train[index]
            is_show_keypoint = True
            
            if kp.pt[0] == ckp.pt[0]  and kp.pt[1] == ckp.pt[1] and kp.size == ckp.size and kp.angle == ckp.angle and kp.response == ckp.response:
                cp_count = cp_count + 1
                
                if is_show_keypoint:
                    print("KeyPoint1: x={}, y={}, size={}, angle={}, response={}, octave={}, class_id={}".format(
                        kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id
                    ))
            if is_show_keypoint and np.array_equal(des2,kp_query):
                print(f"Descriptor same keypoint {index + 1}: {des2[index]}")
        print(f"总特征点{len(kp_train)}")
        print(f"相同的特征点数量：{cp_count}")
        # 打印关键点信息 同一张图片进行2次ORB解析，查看输出内容是否一样 end...

    def compute(queryImg1Path,trainImgPath,isShowImg=False,roi={}):
        start_time = time.time()
        # 创建ORB检测器，设置检测器参数nfeatures=3000 - 特征点最大值 fastThreshold：FAST检测阈值，用于确定一个像素是否是关键点。默认值为20。
        # fastThreshold 通过比较中心像素点和周围一圈像素点的灰度值，快速判断是否为角点。
        # nlevels 较大的 nlevels 会导致更多的图像金字塔层数，允许在更广泛的尺度上进行特
        # 较小的 scaleFactor 会导致金字塔层数增加，从而使得在更多尺度上检测到关键点。这对于处理不同尺寸的特征物体或图像比例变化较大的场景非常有用。
        # edgeThreshold：边界阈值，用于决定图像边界处是否要舍弃特征点。调整这个值可能会影响是否检测到边缘附近的关键点。
        orb = cv.ORB_create(nfeatures=10000,fastThreshold=40,scaleFactor=1.2,nlevels=8,edgeThreshold=31,patchSize=31,WTA_K=2,scoreType=cv.ORB_FAST_SCORE)
        
        # 设置每2x2像素区域内只取3个特征点
        # orb.setPatchSize(2)
        # orb.setMaxFeatures(3)


        queryImg = cv.imread(queryImg1Path, cv.IMREAD_GRAYSCALE)
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



        # 使用BFMatcher进行特征匹配
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # mask：可选参数，用于指定一个掩码，限制哪些匹配将被考虑。如果为 None，则考虑所有匹配。
        matches = bf.match(des_query, des_train)

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
                if match.distance < 50:
                    matching_query_kp.append(kp_query[match.queryIdx])
                    matching_train_kp.append(kp_train[match.trainIdx])
                    new_matches.append(match)
         
            # 获取匹配的关键点下标对照
            query_unique_keys_set = {obj.queryIdx for obj in new_matches}
            train_unique_keys_set = {obj.trainIdx for obj in new_matches}
            print(f"特征点匹配数：{len(query_unique_keys_set)}")

            # 原图-训练图：绘制匹配的关键点
            train_marked = cv.drawKeypoints(trainImgRoi, matching_train_kp, None, color=(0, 255, 0), flags=2)

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

            

            plt.figure(figsize=(6, 12))

            # 显示结果
            if roi != None:
                plt.suptitle(f'ROI:  ({roi["x1"]},{roi["x2"]}) : ({roi["y1"]},{roi["y2"]})', fontsize=16)

            plt.subplot(221), 
            plt.imshow(train_marked),
            plt.title(f'原图 {os.path.basename(queryImg1Path)}（关键点:{len(kp_train)}）  \n\n黄色：{len(non_not_matching_train)} 关键点（不与推理图匹配） \n绿色：{len(matching_query_kp)} （与推理图匹配的关键点）')
            
            
            plt.subplot(222), 
            plt.imshow(query_all_marked), 
            plt.title(f'推理图 {os.path.basename(trainImgPath)}（关键点:{len(kp_query)}）  \n\n绿色: { len(matching_query)}（与原图匹配） \n红色：{ len(non_not_matching_query)} （与原图不匹配）')
           
            plt.subplot(223), 
            plt.imshow(query_marked), 
            plt.title(f'推理图 {os.path.basename(trainImgPath)}（关键点:{len(kp_query)}）   \n绿色：{ len(matching_query)} （与原图匹配的关键点） 匹配率：{round(len(matching_query)/len(kp_query) * 100 ,2)}%' )
            
            plt.subplot(224), 
            plt.imshow(query_not_marked), 
            plt.title(f'推理图 {os.path.basename(trainImgPath)}（关键点:{len(kp_query)}）   \n红色：{ len(non_not_matching_query)} （与原图不匹配的关键点）差异率：{round(len(non_not_matching_query)/len(kp_query) * 100 ,2)}%')

            plt.show()


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

def roi_match(keypoints):
    x1, y1, x2, y2 = 200, 200, 400, 400

class TestImageSURF(unittest.TestCase):

   def test_SURF(self):
       ORB = IMAGE_ORB()
    #    img1Path = r'.\img\camera\img-6-x2.jpg'
    #    img2Path = r'.\img\camera\img-6-x1.jpg'
    
    # img[y1:y2,x1:yx2] 取左上角（750px，520px）到右下角（1500px,780px）区域
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

       test = test_4
      
       
       result = IMAGE_ORB.compute(test["queryImg1Path"], test["trainImgPath"],isShowImg = True,roi=test["roi"])

       self.assertEqual(result, None)

if __name__ == '__main__':
   unittest.main()        
