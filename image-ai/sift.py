import unittest
import cv2 as cv
import time

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
        # kp2, des2 = sift.detectAndCompute(img2,None)

        print(f"耗时: {round(time.time() - start_time,4)} 秒")
        cv.imshow('sift', outImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def SITF_IMG(img1Path,img2Path,isShowImg=False):
        start_time = time.time()
        
        img1 = cv.imread(img1Path)
        img2 = cv.imread(img2Path)

        # 将图片转换为灰度图像
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # 创建SIFT检测器
        # 创建SIFT检测器，设置检测器参数nfeatures=3000 - 特征点最大值 contrastThreshold=0.03 - 对比度阈值
        sift = cv.SIFT_create(nfeatures=3000,contrastThreshold=0.03)
        # 检测并计算图像中的特征点
        kp1 = sift.detect(gray1,None)
        kp2 = sift.detect(gray2,None)


        kp1, des1 = sift.compute(img1, kp1)
        kp2, des2 = sift.compute(img2, kp2)
        bf = cv.BFMatcher(crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        
        # k对最佳匹配
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []

        # 差异性的特征点
        diff_good = []
        
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
            else:
                diff_good.append([m])

        # 计算匹配点数量
        num_matches = len(good)

        # 计算相似度
        similarity = num_matches / len(kp1)

    
        print(f"SIFT特征点数量: img1：{  len(kp1) } img2：{ len(kp2) }")
        print(f"特征点匹配数量: { num_matches }")
        print(f"匹配率: { round(similarity * 100 ,2)  }%")
        print(f"耗时: {round(time.time() - start_time,4)} 秒")

        if isShowImg:
            # img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

            # matches：关键点匹配结果。
            # flags：可选参数，用于控制匹配结果的绘制方式。例如，可以设置 
            # print(cv.DRAW_MATCHES_FLAGS_DEFAULT,cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            outImg = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            # diffImg = cv.drawMatchesKnn(img1, kp1, img2, kp2, diff_good, None, flags=2)

            # 特征点图片
            #show('img3', img3)
            # 对比图片
            show('sift_out', outImg)
            # 差异性图片
            # show('sift_diff_out', diffImg)
            
            cv.destroyAllWindows()

def show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)        

class TestImageSURF(unittest.TestCase):

   def test_SURF(self):
       img1Path = r'.\img\camera\img-7-x1.jpg'
       img2Path = r'.\img\camera\img-7.jpg'
       result = ImageSURF.SITF_IMG(img1Path, img2Path,isShowImg = True)
    #    result = ImageSURF.SITF(img1Path, img2Path)
    #    result = ImageSURF.SURF(img1Path, img2Path)
       self.assertEqual(result, None)

if __name__ == '__main__':
   unittest.main()