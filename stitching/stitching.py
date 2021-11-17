import cv2.cv2 as cv2
import numpy as np


# 检测图像的SIFT关键特征点
def sift_key_points_detect(image):
    # 处理图像一般很少用到彩色信息，通常直接将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取图像特征sift-SIFT特征点,实例化对象sift
    sift = cv2.xfeatures2d.SIFT_create()

    # key_points:特征点向量,向量内的每一个元素是一个KeyPoint对象，包含了特征点的各种属性信息(角度、关键特征点坐标等)
    # features:表示输出的sift特征向量，通常是128维的
    key_points, features = sift.detectAndCompute(image, None)

    # cv2.drawKeyPoints():在图像的关键特征点部位绘制一个小圆圈
    # 如果传递标志flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,它将绘制一个大小为keypoint的圆圈并显示它的方向
    # 这种方法同时显示图像的坐标，size和方向，是最能显示特征的一种绘制方式
    key_points_image = cv2.drawKeypoints(
        gray_image, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 返回带关键特征点的图像、关键特征点和sift的特征向量
    return key_points_image, key_points, features


# 使用KNN检测来自左右图像的SIFT特征，随后进行匹配
def get_feature_point_ensemble(features_right, features_left):
    # 创建BFMatcher对象解决匹配
    bf = cv2.BFMatcher()
    # knnMatch()函数：返回每个特征点的最佳匹配k个匹配点
    # features_right为模板图，features_left为匹配图
    matches = bf.knnMatch(features_right, features_left, k=2)
    # 利用sorted()函数对matches对象进行升序(默认)操作
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    # x:x[]字母可以随意修改，排序方式按照中括号[]里面的维度进行排序，[0]按照第一维排序，[2]按照第三维排序

    # 建立列表good用于存储匹配的点集
    good = []

    for m, n in matches:
        # ratio的值越大，匹配的线条越密集，但错误匹配点也会增多
        ratio = 0.68
        if m.distance < ratio * n.distance:
            good.append(m)

    # 返回匹配的关键特征点集
    return good


# 计算视角变换矩阵H，用H对右图进行变换并返回全景拼接图像
def panorama_stitching(image_right, image_left):
    _, key_points_right, features_right = sift_key_points_detect(image_right)
    _, key_points_left, features_left = sift_key_points_detect(image_left)
    good_match = get_feature_point_ensemble(features_right, features_left)

    # 当筛选项的匹配对大于4对(因为homography单应性矩阵的计算需要至少四个点)时,计算视角变换矩阵
    if len(good_match) > 4:
        # 获取匹配对的点坐标
        ptsR = np.float32(
            [key_points_right[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        ptsL = np.float32(
            [key_points_left[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

        # ransac_threshold：将点对视为内点的最大允许重投影错误阈值(仅用于RANSAC和RHO方法时),若srcPoints和dstPoints是以像素为单位的，该参数通常设置在1到10的范围内
        ransac_threshold = 6

        # cv2.findHomography():计算多个二维点对之间的最优单映射变换矩阵 H(3行x3列),使用最小均方误差或者RANSAC方法
        # 函数作用:利用基于RANSAC的鲁棒算法选择最优的四组配对点，再计算转换矩阵H(3*3)并返回,以便于反向投影错误率达到最小
        homograghy, status = cv2.findHomography(
            ptsR, ptsL, cv2.RANSAC, ransac_threshold)

        # cv2.warpPerspective()：透视变换函数，用于解决cv2.warpAffine()不能处理视场和图像不平行的问题
        # 作用：就是对图像进行透视变换，可保持直线不变形，但是平行线可能不再平行
        panorama = cv2.warpPerspective(
            image_right, homograghy, (image_right.shape[1] + image_left.shape[1], image_right.shape[0]))

        cv2.imshow("扭曲变换后的右图", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 将左图加入到变换后的右图像的左端即获得最终图像
        panorama[0:image_left.shape[0], 0:image_left.shape[1]] = image_left

        # 返回全景拼接的图像
        return panorama


if __name__ == '__main__':
    # 读取需要拼接的图像,需要注意图像左右的顺序
    image_left = cv2.imread("images/1.jpg")
    image_right = cv2.imread("images/2.jpg")

    # 通过调用cv2.resize()使用插值的方式来改变图像的尺寸，保证左右两张图像大小一致
    # cv2.resize()函数中的第二个形参dsize表示输出图像大小尺寸，当设置为0(None)时，则表示按fx与fy与原始图像大小相乘得到输出图像尺寸大小
    image_right = cv2.resize(image_right, None, fx=0.4, fy=0.24)
    image_left = cv2.resize(image_left, (image_right.shape[1], image_right.shape[0]))

    # 获取检测到关键特征点后的图像的相关参数
    key_points_image_right, key_points_right, features_right = sift_key_points_detect(image_right)
    key_points_image_left, key_points_left, features_left = sift_key_points_detect(image_left)

    # 利用np.hstack()函数同时将原图和绘有关键特征点的图像沿着竖直方向(水平顺序)堆叠起来
    cv2.imshow("左图关键特征点检测", np.hstack((image_left, key_points_image_left)))
    # 一般在imshow后设置 waitKey(0) , 代表按任意键继续
    cv2.waitKey(0)
    # 删除先前建立的窗口
    cv2.destroyAllWindows()
    cv2.imshow("右图关键特征点检测", np.hstack((image_right, key_points_image_right)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    good_match = get_feature_point_ensemble(features_right, features_left)

    # cv2.drawMatches():在提取两幅图像特征之后，画出匹配点对连线
    # matchColor – 匹配的颜色（特征点和连线),若matchColor==Scalar::all(-1),颜色随机
    all_good_match_image = cv2.drawMatches(
        image_right, key_points_right, image_left, key_points_left, good_match, None, None, None, None, flags=2)
    cv2.imshow("所有匹配的SIFT关键特征点连线", all_good_match_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 把图片拼接成全景图并保存
    panorama = panorama_stitching(image_right, image_left)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result", panorama)
    cv2.imwrite("res.jpg", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
