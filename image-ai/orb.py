import unittest
import cv2 as cv
import numpy as np
import time
import os
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import Button

# 替换成你选择的中文字体路径
SHOW_IMG=True

PLT_IMG_LIST=[]
PLT_COL = 3
PLT_ROW = 3
PLT_CURRENT_INDEX=0


if SHOW_IMG:
    font_path = r'd:\msyh.ttc'
    if os.path.exists(font_path):
        # 设置中文字体
        chinese_font = FontProperties(fname=font_path)
        # 使用中文字体
        plt.rcParams['font.family'] = chinese_font.get_name()

# 按钮点击事件的回调函数
def on_prev_click(event):
    global PLT_IMG_LIST
    global PLT_CURRENT_INDEX
    
    print("on_prev_click")
    if PLT_CURRENT_INDEX - 1 >= 0:
        PLT_CURRENT_INDEX = PLT_CURRENT_INDEX - 1
        plt_show()
  

def on_next_click(event):
    global PLT_IMG_LIST
    global PLT_CURRENT_INDEX
    global PLT_COL
    global PLT_ROW

    print("on_next_click")
    if PLT_CURRENT_INDEX + 1 < len(PLT_IMG_LIST) / (PLT_COL * PLT_ROW):
        PLT_CURRENT_INDEX = PLT_CURRENT_INDEX + 1
        plt_show()
    

plt.figure(figsize=(16, 16))
# 调整子图之间的垂直间距
plt.subplots_adjust(hspace=0.5)

# 添加按钮的坐标轴
ax_prev = plt.axes([0.4, 0.015, 0.08, 0.055])
ax_next = plt.axes([0.55, 0.015, 0.08, 0.055])

# 创建按钮
btn_prev = Button(ax_prev, 'Previous')
btn_next = Button(ax_next, 'Next')

# 将回调函数连接到按钮
btn_prev.on_clicked(on_prev_click)
btn_next.on_clicked(on_next_click)

class IMAGE_ORB():
    def computeAndShow(queryImgPath,trainImgPath,isShowImg=False,roi={},subplot=None, rows =1 ,cols = 1):
        global PLT_IMG_LIST
        global PLT_COL
        global PLT_ROW

        start_time = time.time()
        # 创建ORB检测器，设置检测器参数nfeatures=3000 - 特征点最大值 fastThreshold：FAST检测阈值，用于确定一个像素是否是关键点。默认值为20。
        # fastThreshold 通过比较中心像素点和周围一圈像素点的灰度值，快速判断是否为角点。
        # nlevels 较大的 nlevels 会导致更多的图像金字塔层数，允许在更广泛的尺度上进行特
        # 较小的 scaleFactor 会导致金字塔层数增加，从而使得在更多尺度上检测到关键点。这对于处理不同尺寸的特征物体或图像比例变化较大的场景非常有用。
        # edgeThreshold：边界阈值，用于决定图像边界处是否要舍弃特征点。调整这个值可能会影响是否检测到边缘附近的关键点。

        orb = cv.ORB_create(nfeatures=10000,fastThreshold=20,scaleFactor=1.2,nlevels=2,edgeThreshold=15,patchSize=15,WTA_K=4,scoreType=cv.ORB_FAST_SCORE)
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
                train_marked_title = f'\n原图 {os.path.basename(trainImgPath)}（关键点:{len(kp_train)}）  \n黄色：{len(non_not_matching_train)} 关键点（不与推理图匹配） \n绿色：{len(matching_query_kp)} （与推理图匹配的关键点）'

                plt.subplot(121), 
                plt.imshow(train_marked),
                plt.title(train_marked_title)

    
                query_title = f'\n推理图 {os.path.basename(queryImgPath)}（关键点:{len(kp_query)}）  \n绿色: { len(matching_query)}（与原图匹配） 匹配率：{round(len(matching_query)/len(kp_query) * 100 ,2)}%   红色：{ len(non_not_matching_query)} （与原图不匹配）差异率：{round(len(non_not_matching_query)/len(kp_query) * 100 ,2)}%'
                plt.subplot(122), 
                plt.imshow(query_all_marked), 
                plt.title(query_title)
                
            else:
                PLT_COL = subplot[1]
                PLT_ROW = subplot[0]

                train_marked_title = f'\n原图 {os.path.basename(trainImgPath)}（关键点:{len(kp_train)}）  \n黄色：{len(non_not_matching_train)} 关键点（不与推理图匹配） \n绿色：{len(matching_query_kp)} （与推理图匹配的关键点）'
                # plt.subplot(subplot[0],subplot[1],subplot[2]), 
                # plt.imshow(train_marked),
                # plt.title(train_marked_title)
                
                # 存入缓存
                PLT_IMG_LIST.append({
                    "title": train_marked_title,
                    "img": train_marked
                })
                
                query_title = f'\n推理图 {os.path.basename(queryImgPath)}（关键点:{len(kp_query)}）  \n绿色: { len(matching_query)}（与原图匹配） 匹配率：{round(len(matching_query)/len(kp_query) * 100 ,2)}%   \n红色：{ len(non_not_matching_query)} （与原图不匹配）差异率：{round(len(non_not_matching_query)/len(kp_query) * 100 ,2)}%'
         
                # plt.subplot(subplot[0],subplot[1],int(subplot[2]) + 1), 
                # plt.imshow(query_all_marked), 
                # plt.title(query_title)

                PLT_IMG_LIST.append({
                    "title": query_title,
                    "img": query_all_marked
                })

                
                # 推理图（彩色原图）
                # plt.subplot(subplot[0],subplot[1],int(subplot[2]) + 2), 
                # plt.imshow(cv.imread(queryImgPath)),
                # plt.title(queryImgPath)

                # 推理图（彩色原图）
                PLT_IMG_LIST.append({
                    "title": os.path.basename(queryImgPath),
                    "img": cv.cvtColor(cv.imread(queryImgPath), cv.COLOR_BGR2RGB)
                })
                

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
        # 创建ORB检测器，设置检测器参数nfeatures=3000 - 特征点最大值 fastThreshold：FAST检测阈值，用于确定一个像素是否是关键点。默认值为20。
        # fastThreshold 通过比较中心像素点和周围一圈像素点的灰度值，快速判断是否为角点。
        # nlevels 较大的 nlevels 会导致更多的图像金字塔层数，允许在更广泛的尺度上进行特
        # 较小的 scaleFactor 会导致金字塔层数增加，从而使得在更多尺度上检测到关键点。这对于处理不同尺寸的特征物体或图像比例变化较大的场景非常有用。
        # edgeThreshold：边界阈值，用于决定图像边界处是否要舍弃特征点。调整这个值可能会影响是否检测到边缘附近的关键点。
        # 通过设置 nLevels 参数为1，可以减少图像金字塔的层数，从而限制旋转不变性
        orb = cv.ORB_create(nfeatures=10000,fastThreshold=80,scaleFactor=1.2,nlevels=1,edgeThreshold=15,patchSize=15,WTA_K=4,scoreType=cv.ORB_FAST_SCORE)
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
        new_matches = []
        # match.distance 表示两个特征描述子之间的距离，通常情况下，距离越小表示两个特征越相似
        for match in matches:
            if match.distance < 1000:
                new_matches.append(match)
        
        # 获取匹配的关键点下标对照
        train_unique_keys_set = {obj.trainIdx for obj in new_matches}
        query_unique_keys_set = {obj.queryIdx for obj in new_matches}

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
            "queryDiffKpNum":query_diff_num,
            "querySameRate": round( query_same_num / kp_query_num, 4 ),
            "queryDiffRate": round( query_diff_num / kp_query_num, 4 ),
            "trainKpNum":kp_train_num,
            "trainSameKpNum":train_same_num,
            "trainDiffKpNum":train_diff_num,
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

        test_12 = {
            "baseImg":r"C:\Users\17154\Desktop\MVS\12-15\base_img.jpg",
            "targetImg1":r"C:\Users\17154\Desktop\MVS\12-15\img-10-03-49-67361.jpg",
            "targetImg2":r"C:\Users\17154\Desktop\MVS\12-15\img-10-07-13-54488.jpg",
            "targetImg3":r"C:\Users\17154\Desktop\MVS\12-15\img-10-05-15-29897.jpg",
            "roi":None
        }

    
        test_13 = {
            "baseImg":r"C:\Users\17154\Desktop\MVS\12-19-test\base_img.jpg",
            "targetImg1":r"C:\Users\17154\Desktop\MVS\12-19-test\base_img_1.jpg",
            "targetImg2":r"C:\Users\17154\Desktop\MVS\12-19-test\img-09-31-01-27501.jpg",
            "targetImg3":r"C:\Users\17154\Desktop\MVS\12-19-test\base_img.jpg",
            "roi":None
        }

        test_14 = {
            "baseImg":r"C:\Users\17154\Desktop\MVS\12-19-test\before\base_img.jpg",
            "targetImg1":r"C:\Users\17154\Desktop\MVS\12-19-test\before\img-10-31-07-15188.jpg",
            "targetImg2":r"C:\Users\17154\Desktop\MVS\12-19-test\before\img-10-31-52-71705.jpg",
            "targetImg3":r"C:\Users\17154\Desktop\MVS\12-19-test\before\img-10-32-31-78894.jpg",
            "targetImg4":r"C:\Users\17154\Desktop\MVS\12-19-test\before\img-10-32-44-88144.jpg",
            "targetImg5":r"C:\Users\17154\Desktop\MVS\12-19-test\before\img-10-43-08-97453.jpg",
            "targetImg6":r"C:\Users\17154\Desktop\MVS\12-19-test\before\img-11-00-32-11102.jpg",
            "targetImg7":r"C:\Users\17154\Desktop\MVS\12-19-test\before\img-11-23-52-96339.jpg",
            "targetImg8":r"C:\Users\17154\Desktop\MVS\12-19-test\before\img-11-27-42-55504.jpg",
            "roi":None
        }

        yan_jiang = {
            "baseImg":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_16\baseImg\before\base_img.jpg",
            "targetImg1":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_20\photo\12-21\img-10-17-33-65414.jpg",
            "targetImg2":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_20\photo\12-21\img-10-18-14-77025.jpg",
            "targetImg3":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_20\photo\12-21\img-10-19-21-57876.jpg",
            "targetImg4":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_16\alarm\before\12-21\img-10-15-14-42017.jpg",
            "targetImg5":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_16\alarm\before\12-21\img-10-15-01-21505.jpg",
            "targetImg6":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_16\alarm\before\12-21\img-10-15-18-64865.jpg",
            "roi":None
        }

        yan_jiang_2={
            "baseImg1":r"C:\Users\17154\Desktop\MVS\test\1\base_img.jpg",
            "baseImg2":r"C:\Users\17154\Desktop\MVS\test\1\base_img_1.jpg",
            "baseImg3":r"C:\Users\17154\Desktop\MVS\test\1\base_img_2.jpg",
            "targetImg1":r"C:\Users\17154\Desktop\MVS\test\1\img-1-1.jpg",
            "targetImg2":r"C:\Users\17154\Desktop\MVS\test\1\img-1-2.jpg",
            "targetImg3":r"C:\Users\17154\Desktop\MVS\test\1\img-1-3.jpg",
            "targetImg4":r"C:\Users\17154\Desktop\MVS\test\1\img-2-1.jpg",
            "targetImg5":r"C:\Users\17154\Desktop\MVS\test\1\img-2-2.jpg",
            "targetImg6":r"C:\Users\17154\Desktop\MVS\test\1\img-2-3.jpg",
            "roi":None

        }

        office={
            "baseImg1":r"C:\Users\17154\MVS\Data\img1.jpg",
            "baseImg2":r"C:\Users\17154\MVS\Data\img1.jpg",
            "baseImg3":r"C:\Users\17154\MVS\Data\img1.jpg",
            "targetImg1":r"C:\Users\17154\MVS\Data\img1.jpg",
            "targetImg2":r"C:\Users\17154\MVS\Data\img1.jpg",
            "targetImg3":r"C:\Users\17154\MVS\Data\img1.jpg",
            "targetImg4":r"C:\Users\17154\MVS\Data\img1.jpg",
            "targetImg5":r"C:\Users\17154\MVS\Data\img1.jpg",
            "targetImg6":r"C:\Users\17154\MVS\Data\img1.jpg",
            "roi":None

        }

        IS_SHOW_IMG = True
        result = IMAGE_ORB.computeAndShow(yan_jiang_2["baseImg2"], yan_jiang_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,1], rows = 0 ,cols = 0)
        result = IMAGE_ORB.computeAndShow(yan_jiang_2["baseImg3"], yan_jiang_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,4], rows = 0 ,cols = 0)

        result = IMAGE_ORB.computeAndShow(yan_jiang_2["targetImg1"], yan_jiang_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,7], rows = 0 ,cols = 0)
        result = IMAGE_ORB.computeAndShow(yan_jiang_2["targetImg2"], yan_jiang_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,10], rows = 0 ,cols = 0)

        result = IMAGE_ORB.computeAndShow(yan_jiang_2["targetImg4"], yan_jiang_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

        if IS_SHOW_IMG:
            plt_show()
        self.assertEqual(result, None)

def plt_show():
    global PLT_IMG_LIST
    global PLT_CURRENT_INDEX
    global PLT_ROW
    global PLT_COL
    
    start_index = int(PLT_CURRENT_INDEX * PLT_ROW * PLT_COL)
    end_index = int(start_index) + int(PLT_ROW * PLT_COL)

    for i in range(start_index,end_index):
        current_index = i - start_index
        if i < len(PLT_IMG_LIST):
            value = PLT_IMG_LIST[i]
            plt.subplot(PLT_ROW,PLT_COL,int(current_index + 1)), 
            plt.imshow(value["img"]),
            plt.title(value["title"])


    plt.draw()
    plt.show()

# 验证阳江现场图片
def yan_jiang_test():
    yan_jiang = {
        "baseImg":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_16\baseImg\before\base_img.jpg",
        "photo-1":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_20\photo\12-21\img-10-17-33-65414.jpg",
        "photo-2":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_20\photo\12-21\img-10-18-14-77025.jpg",
        "photo-3":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_20\photo\12-21\img-10-19-21-57876.jpg",
        "targetImg1":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_16\alarm\before\12-21\img-10-15-14-42017.jpg",
        "targetImg2":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_16\alarm\before\12-21\img-10-15-01-21505.jpg",
        "targetImg3":r"C:\Users\17154\Desktop\MVS\yan_jiang\10_16\alarm\before\12-21\img-10-15-18-64865.jpg",
        "roi":None
    }

    yan_jiang_2={
        "baseImg":r"C:\Users\17154\Desktop\MVS\test\1\base_img.jpg",
        "baseImg2":r"C:\Users\17154\Desktop\MVS\test\1\base_img_1.jpg",
        "baseImg3":r"C:\Users\17154\Desktop\MVS\test\1\base_img_2.jpg",
        "targetImg1":r"C:\Users\17154\Desktop\MVS\test\1\img-1-1.jpg",
        "targetImg2":r"C:\Users\17154\Desktop\MVS\test\1\img-1-2.jpg",
        "targetImg3":r"C:\Users\17154\Desktop\MVS\test\1\img-1-3.jpg",
        "targetImg4":r"C:\Users\17154\Desktop\MVS\test\1\img-2-1.jpg",
        "targetImg5":r"C:\Users\17154\Desktop\MVS\test\1\img-2-2.jpg",
        "targetImg6":r"C:\Users\17154\Desktop\MVS\test\1\img-2-3.jpg",
        "roi":None

    }   

    IS_SHOW_IMG = True
    IMAGE_ORB.computeAndShow(yan_jiang["photo-1"], yan_jiang["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang["photo-2"], yan_jiang["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang["targetImg1"], yan_jiang["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(yan_jiang["photo-3"], yan_jiang["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang["targetImg2"], yan_jiang["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang["targetImg3"], yan_jiang["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(yan_jiang_2["baseImg2"], yan_jiang_2["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang_2["targetImg1"], yan_jiang_2["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang_2["targetImg4"], yan_jiang_2["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(yan_jiang_2["baseImg2"], yan_jiang_2["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang_2["targetImg2"], yan_jiang_2["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang_2["targetImg3"], yan_jiang_2["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(yan_jiang_2["baseImg3"], yan_jiang_2["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang_2["targetImg5"], yan_jiang_2["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(yan_jiang_2["targetImg6"], yan_jiang_2["baseImg"],isShowImg = IS_SHOW_IMG,roi=yan_jiang_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    if IS_SHOW_IMG:
        plt_show()

# 验证办公室对比
def office_test():
    ORB = IMAGE_ORB()
    office_1={
        "baseImg1":r"C:\Users\17154\MVS\Data\img-1.jpg",
        "baseImg2":r"C:\Users\17154\MVS\Data\img-1-1.jpg",
        "baseImg3":r"C:\Users\17154\MVS\Data\img-1-2.jpg",
        "baseImg4":r"C:\Users\17154\MVS\Data\img-1-3.jpg",
        "x1":r"C:\Users\17154\MVS\Data\img-1-x1.jpg",
        "x1-1":r"C:\Users\17154\MVS\Data\img-1-x1-1.jpg",
        "x1-2":r"C:\Users\17154\MVS\Data\img-1-x1-2.jpg",
        "x2":r"C:\Users\17154\MVS\Data\img-1-x2.jpg",
        "x2-1":r"C:\Users\17154\MVS\Data\img-1-x2-1.jpg",
        "x2-2":r"C:\Users\17154\MVS\Data\img-1-x2-2.jpg",
        "roi" : {
                "x1":300,
                "y1":300,
                "x2":1100,
                "y2":800,
            },  
        
    }
    office_2={
        "baseImg1":r"C:\Users\17154\MVS\Data\img-2.jpg",
        "baseImg2":r"C:\Users\17154\MVS\Data\img-2-1.jpg",
        "baseImg3":r"C:\Users\17154\MVS\Data\img-2-2.jpg",
        "baseImg4":r"C:\Users\17154\MVS\Data\img-2-3.jpg",
        "x1":r"C:\Users\17154\MVS\Data\img-2-x1.jpg",
        "x1-1":r"C:\Users\17154\MVS\Data\img-2-x1-1.jpg",
        "x1-2":r"C:\Users\17154\MVS\Data\img-2-x1-2.jpg",
        "x2":r"C:\Users\17154\MVS\Data\img-2-x2.jpg",
        "x2-1":r"C:\Users\17154\MVS\Data\img-2-x2-1.jpg",
        "x2-2":r"C:\Users\17154\MVS\Data\img-2-x2-2.jpg",
        "roi" : {
                "x1":900,
                "y1":1200,
                "x2":2700,
                "y2":2400,
            },
    }
    IS_SHOW_IMG = True
    IMAGE_ORB.computeAndShow(office_1["baseImg2"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_1["baseImg3"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_1["baseImg4"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_1["x1"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_1["x1-1"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_1["x1-2"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_1["x2"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_1["x2-1"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_1["x2-2"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(office_2["baseImg2"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_2["baseImg3"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_2["baseImg4"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_2["x1"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_2["x1-1"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_2["x1-2"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_2["x2"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_2["x2-1"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_2["x2-2"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)


    IMAGE_ORB.computeAndShow(office_1["x2"], office_1["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_1["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_2["x1"], office_2["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_2["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    if IS_SHOW_IMG:
        plt_show()
    

# 验证办公室对比
def office_test_2():
    ORB = IMAGE_ORB()
    office_3={
        "baseImg1":r"C:\Users\17154\MVS\Data\img3.jpg",
        "baseImg2":r"C:\Users\17154\MVS\Data\img3-1.jpg",
        "baseImg3":r"C:\Users\17154\MVS\Data\img3-2.jpg",
        "x1":r"C:\Users\17154\MVS\Data\img3-x1.jpg",
        "x1-1":r"C:\Users\17154\MVS\Data\img3-x1-1.jpg",
        "x1-2":r"C:\Users\17154\MVS\Data\img3-x1-2.jpg",
        "x2":r"C:\Users\17154\MVS\Data\img3-x2.jpg",
        "x2-1":r"C:\Users\17154\MVS\Data\img3-x2-1.jpg",
        "x2-2":r"C:\Users\17154\MVS\Data\img3-x2-2.jpg",
        "x3":r"C:\Users\17154\MVS\Data\img3-x3.jpg",
        "x3-1":r"C:\Users\17154\MVS\Data\img3-x3-1.jpg",
        "roi" : {
                "x1":0,
                "y1":50,
                "x2":1800,
                "y2":1100,
            },  
        
    }
    office_4={
        "baseImg1":r"C:\Users\17154\MVS\Data\img4.jpg",
        "baseImg2":r"C:\Users\17154\MVS\Data\img4-1.jpg",
        "baseImg3":r"C:\Users\17154\MVS\Data\img4-2.jpg",
        "x1":r"C:\Users\17154\MVS\Data\img4-x1.jpg",
        "x1-1":r"C:\Users\17154\MVS\Data\img4-x1-1.jpg",
        "x1-2":r"C:\Users\17154\MVS\Data\img4-x1-2.jpg",
        "x2":r"C:\Users\17154\MVS\Data\img4-x2.jpg",
        "x2-1":r"C:\Users\17154\MVS\Data\img4-x2-1.jpg",
        "x2-2":r"C:\Users\17154\MVS\Data\img4-x2-2.jpg",
        "x3":r"C:\Users\17154\MVS\Data\img4-x3.jpg",
        "x3-1":r"C:\Users\17154\MVS\Data\img4-x3-1.jpg",
        "roi" : {
                "x1":350,
                "y1":700,
                "x2":2900,
                "y2":2800,
            },  
    }
    IS_SHOW_IMG = True
    # IMAGE_ORB.computeAndShow(office_3["baseImg2"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_3["baseImg3"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    # IMAGE_ORB.computeAndShow(office_3["baseImg3"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_3["x1"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_3["x2"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_3["x3"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(office_4["baseImg2"], office_4["x2-1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_4["baseImg3"], office_4["x2-1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_4["baseImg3"], office_4["x2-1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_4["x1"], office_4["x2-1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_4["x2"], office_4["x2-1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_4["x3"], office_4["x2-1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)


    if IS_SHOW_IMG:
        plt_show()

# 验证曝光时间不同，以及放异物
def test_light():
    ORB = IMAGE_ORB()
    IS_SHOW_IMG = True
    folder_path=r"C:\Users\17154\MVS\Data\night"
    # 使用os.listdir获取文件夹中的文件列表
    files = os.listdir(folder_path)

    light_roi =  {
                "x1":100,
                "y1":100,
                "x2":1100,
                "y2":1100,
            }
    
    # 打印文件列表
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            input_string = file_path
            # 找到第一个横杠的位置
            index_of_first_hyphen = input_string.find('-')

            # 如果找到了第一个横杠，则在第一个横杠之后找第二个横杠的位置
            if index_of_first_hyphen != -1:
                index_of_second_hyphen = input_string.find('-', index_of_first_hyphen + 1)

                # 如果找到了第二个横杠，则获取第二个横杠之前的子字符串
                if index_of_second_hyphen != -1:
                    file_prefix = input_string[:index_of_second_hyphen]
                    result = file_prefix + "-1.bmp"
                    print(file_path,result)
                    if result == file_path:
                        continue

                    
                    IMAGE_ORB.computeAndShow(file_path, result,isShowImg = IS_SHOW_IMG,roi=light_roi,subplot=[4,3,13], rows = 0 ,cols = 0)
    if IS_SHOW_IMG:
        plt_show()


# 验证曝光时间不同，以及放异物
def office_test_3():
    ORB = IMAGE_ORB()
    office_3={
        "baseImg1":r"C:\Users\17154\MVS\Data\office_3\img-1-1.jpg",
        "baseImg2":r"C:\Users\17154\MVS\Data\office_3\img-1-2.jpg",
        "baseImg3":r"C:\Users\17154\MVS\Data\office_3\img-1-3.jpg",
        "x1-1":r"C:\Users\17154\MVS\Data\office_3\img-1-x1-1.jpg",
        "x1-2":r"C:\Users\17154\MVS\Data\office_3\img-1-x1-2.jpg",
        "x2-1":r"C:\Users\17154\MVS\Data\office_3\img-1-x2-1.jpg",
        "x2-2":r"C:\Users\17154\MVS\Data\office_3\img-1-x2-2.jpg",
        "roi" : {
            "x1":200,
            "y1":100,
            "x2":1100,
            "y2":1100,
        },  

    }
    office_4={
       "baseImg1":r"C:\Users\17154\MVS\Data\office_3\img-2-1.jpg",
        "baseImg2":r"C:\Users\17154\MVS\Data\office_3\img-2-2.jpg",
        "baseImg3":r"C:\Users\17154\MVS\Data\office_3\img-2-3.jpg",
        "x1-1":r"C:\Users\17154\MVS\Data\office_3\img-2-x1-1.jpg",
        "x1-2":r"C:\Users\17154\MVS\Data\office_3\img-2-x1-2.jpg",
        "x2-1":r"C:\Users\17154\MVS\Data\office_3\img-2-x2-1.jpg",
        "x2-2":r"C:\Users\17154\MVS\Data\office_3\img-2-x2-2.jpg",
    "roi" : {
            "x1":50,
            "y1":600,
            "x2":3000,
            "y2":3200,
        },  
    }

    
    office_5={
        "baseImg1":r"C:\Users\17154\MVS\Data\office_3\img-3-1.jpg",
        "baseImg2":r"C:\Users\17154\MVS\Data\office_3\img-3-2.jpg",
        "baseImg3":r"C:\Users\17154\MVS\Data\office_3\img-3-3.jpg",
        "x1-1":r"C:\Users\17154\MVS\Data\office_3\img-3-x1-1.jpg",
        "x1-2":r"C:\Users\17154\MVS\Data\office_3\img-3-x1-2.jpg",
        "x2-1":r"C:\Users\17154\MVS\Data\office_3\img-3-x2-1.jpg",
        "x2-2":r"C:\Users\17154\MVS\Data\office_3\img-3-x2-2.jpg",
        "roi" : {
            "x1":200,
            "y1":100,
            "x2":1100,
            "y2":1100,
        },  

    }

    IS_SHOW_IMG = True

    IMAGE_ORB.computeAndShow(office_5["baseImg2"], office_5["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_5["baseImg3"], office_5["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_5["x1-1"], office_5["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_5["x1-2"], office_5["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(office_5["baseImg2"], office_5["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_5["baseImg3"], office_5["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_5["x2-1"], office_5["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_5["x2-2"], office_5["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)


    IMAGE_ORB.computeAndShow(office_3["baseImg2"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_3["baseImg3"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_3["x1-1"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_3["x2-1"], office_3["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_3["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(office_4["baseImg2"], office_4["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_4["baseImg3"], office_4["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_4["x1-1"], office_4["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(office_4["x2-1"], office_4["baseImg1"],isShowImg = IS_SHOW_IMG,roi=office_4["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)     

    if IS_SHOW_IMG:
        plt_show()
# 验证曝光时间不同
def test_ligth_diff():
    ORB = IMAGE_ORB()
    IS_SHOW_IMG = True
    folder_path=r"C:\Users\17154\MVS\Data\night"
    # 使用os.listdir获取文件夹中的文件列表
    files = os.listdir(folder_path)

    light_roi =  {
                "x1":100,
                "y1":100,
                "x2":1100,
                "y2":1100,
            }
    
    
    light_roi =  {
                "x1":200,
                "y1":200,
                "x2":1000,
                "y2":1000,
            }
    
    light_diff = {
        "img-800":r"C:\Users\17154\MVS\Data\night\test-800-1-1.bmp",
        "img-1000":r"C:\Users\17154\MVS\Data\night\test-1000-1-1.bmp",
        "img-1000-1":r"C:\Users\17154\MVS\Data\night\test-1000-1.bmp",
        "img-1000-2":r"C:\Users\17154\MVS\Data\night\test-1000-2.bmp",
        "img-1400":r"C:\Users\17154\MVS\Data\night\test-1400-1.bmp",
        "img-1800":r"C:\Users\17154\MVS\Data\night\test-1800-1.bmp",
        "img-1400-2":r"C:\Users\17154\MVS\Data\night\test-1400-1-1.bmp",
        "img-1800-2":r"C:\Users\17154\MVS\Data\night\test-1800-1-1.bmp"
    }

                    
    IMAGE_ORB.computeAndShow(light_diff["img-800"], light_diff["img-1000"],isShowImg = IS_SHOW_IMG,roi=light_roi,subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(light_diff["img-1000-1"], light_diff["img-1000"],isShowImg = IS_SHOW_IMG,roi=light_roi,subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(light_diff["img-1400"], light_diff["img-1000"],isShowImg = IS_SHOW_IMG,roi=light_roi,subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(light_diff["img-1800"], light_diff["img-1000"],isShowImg = IS_SHOW_IMG,roi=light_roi,subplot=[4,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(light_diff["img-1400"], light_diff["img-1000"],isShowImg = IS_SHOW_IMG,roi=light_roi,subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(light_diff["img-1400-2"], light_diff["img-1000"],isShowImg = IS_SHOW_IMG,roi=light_roi,subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(light_diff["img-1800"], light_diff["img-1000"],isShowImg = IS_SHOW_IMG,roi=light_roi,subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(light_diff["img-1800-2"], light_diff["img-1000"],isShowImg = IS_SHOW_IMG,roi=light_roi,subplot=[4,3,13], rows = 0 ,cols = 0)

  
    if IS_SHOW_IMG:
        plt_show()


def office_test_4():
    ORB = IMAGE_ORB()
    camera={
        "baseImg1":r"E:\ai_error_img\offfice\test_4\img1-1.jpg",
        "baseImg2":r"E:\ai_error_img\offfice\test_4\img1-2.jpg",
        "baseImg3":r"E:\ai_error_img\offfice\test_4\img1-3.jpg",
        "x1-1":r"E:\ai_error_img\offfice\test_4\img1-x1-1.jpg",
        "x1-2":r"E:\ai_error_img\offfice\test_4\img1-x1-2.jpg",
        "x2-1":r"E:\ai_error_img\offfice\test_4\img1-x2-1.jpg",
        "x2-2":r"E:\ai_error_img\offfice\test_4\img1-x2-2.jpg",
        "x3-1":r"E:\ai_error_img\offfice\test_4\img1-x3-1.jpg",
        "x3-2":r"E:\ai_error_img\offfice\test_4\img1-x3-2.jpg",
        "roi" : None,  

    }
    phone={
       "baseImg1":r"E:\ai_error_img\offfice\test_4\img5-1.jpg",
        "baseImg2":r"E:\ai_error_img\offfice\test_4\img5-2.jpg",
        "baseImg3":r"E:\ai_error_img\offfice\test_4\img5-3.jpg",
        "x1-1":r"E:\ai_error_img\offfice\test_4\img5-x1-1.jpg",
        "x1-2":r"E:\ai_error_img\offfice\test_4\img5-x1-2.jpg",
        "x2-1":r"E:\ai_error_img\offfice\test_4\img5-x2-1.jpg",
        "x2-2":r"E:\ai_error_img\offfice\test_4\img5-x2-2.jpg",
        "x3-1":r"E:\ai_error_img\offfice\test_4\img5-x3-1.jpg",
        "x3-2":r"E:\ai_error_img\offfice\test_4\img5-x3-2.jpg",
        "roi" : {
                "x1":500,
                "y1":600,
                "x2":4000,
                "y2":2500,
        }  
    }



    IS_SHOW_IMG = True

    IMAGE_ORB.computeAndShow(camera["baseImg2"], camera["baseImg1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(camera["x1-1"], camera["baseImg1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(camera["x2-1"], camera["baseImg1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(camera["x3-1"], camera["baseImg1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(phone["baseImg2"], phone["baseImg1"],isShowImg = IS_SHOW_IMG,roi=phone["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(phone["x1-1"], phone["baseImg1"],isShowImg = IS_SHOW_IMG,roi=phone["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(phone["x2-1"], phone["baseImg1"],isShowImg = IS_SHOW_IMG,roi=phone["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(phone["x3-1"], phone["baseImg1"],isShowImg = IS_SHOW_IMG,roi=phone["roi"],subplot=[4,3,13], rows = 0 ,cols = 0)

  
    if IS_SHOW_IMG:
        plt_show()


def diff_time():
    ORB = IMAGE_ORB()
    camera={
        "baseImg1":r"E:\ai_error_img\offfice\test_4\img1-1.jpg",
        "baseImg2":r"E:\ai_error_img\offfice\test_4\img1-2.jpg",
        "baseImg3":r"E:\ai_error_img\offfice\test_4\img1-3.jpg",
        "x3-1":r"E:\ai_error_img\offfice\test_4\img1-x3-1.jpg",
        "x3-2":r"E:\ai_error_img\offfice\test_4\img1-x3-2.jpg",
        "x3-3":r"E:\ai_error_img\offfice\test_4\img1-x3-3.jpg",
        "x3-4":r"E:\ai_error_img\offfice\test_4\img1-x3-4.jpg",
        "x3-14-1":r"E:\ai_error_img\offfice\test_4\img1-x3-14_13.jpg",
        "x3-14-2":r"E:\ai_error_img\offfice\test_4\img1-x3-14_13-2.jpg",
        "x3-14-x1":r"E:\ai_error_img\offfice\test_4\img1-x3-14_13-x1.jpg",
        "x3-14-x2":r"E:\ai_error_img\offfice\test_4\img1-x3-14_13-x2.jpg",
        "roi" : None,  
    }



    IS_SHOW_IMG = True

    IMAGE_ORB.computeAndShow(camera["x3-2"], camera["x3-1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(camera["x3-14-1"], camera["x3-1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(camera["x3-14-2"], camera["x3-1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    IMAGE_ORB.computeAndShow(camera["x3-3"], camera["x3-1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(camera["x3-14-x1"], camera["x3-1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)
    IMAGE_ORB.computeAndShow(camera["x3-14-x2"], camera["x3-1"],isShowImg = IS_SHOW_IMG,roi=camera["roi"],subplot=[3,3,13], rows = 0 ,cols = 0)

    if IS_SHOW_IMG:
        plt_show()

if __name__ == '__main__':
    # 验证办阳江现场图片
    # yan_jiang_test()                     # 环境太多关键点，模具残留物的关键地取值与总关键点比例 占比低

    # 验证办公室对比1
    # office_test()                        # 像素高的相机，对异物的关键点取值更多  （低曝光时间，摄像机关键点取值很少）

    # 验证办公室对比2
    # office_test_2()                      # 高像素的不平整的模具，关键点取值过多

    # 验证办公室对比3
    # office_test_3()                        # 模具复杂的话，异物面积越小，不匹配关键点占比越小

    # 验证手机和工业相机识别相同物体的差异     #像素高的相机，关键点取值更敏感,对小物体残留物关键点更多
    # office_test_4()

    # 验证曝光时间不同，以及放异物区别         #底图要取最近拍摄的图片，不同时间去拍摄图片，图片内容相同， 但不匹配的关键点会出现突增
    # test_light()

    # 验证曝光时间不同                        #曝光时间，图片更清晰，关键点取值更大
    # test_ligth_diff()                   

    diff_time()                           #不用时间点拍摄的相同图片，不匹配点会出现突增

#    unittest.main()        


