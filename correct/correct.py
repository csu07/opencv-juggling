import cv2.cv2 as cv2
import numpy as np


def image_correct(image_data):
    """
    透视变换
    :param image_data:
    :return:
    """
    target_points = [[600, 510], [2380, 470], [2765, 3170], [210, 3190]]  # 获取四个角点A，B，C，D

    height = image_data.shape[0]
    weight = image_data.shape[1]
    four_points = np.array(((0, 0),
                            (weight - 1, 0),
                            (weight - 1, height - 1),
                            (0, height - 1)),
                           np.float32)
    target_points = np.array(target_points, np.float32)  # 统一格式
    m = cv2.getPerspectiveTransform(target_points, four_points)  # 透视变换
    res = cv2.warpPerspective(image_data, m, (weight, height))
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.imwrite("res.jpg", res)


def image_correct1(img_data):
    """
    透视变换：
    1. 先找到目标近似轮廓
    2. 根据近似轮廓顶点坐标进行透视变换
    :param img_data:
    :return:
    """
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯模糊去噪点
    mid_gray = cv2.medianBlur(gray, 3)  # 中值滤波忽略较高阶灰度和较低阶灰度
    cv2.imshow("mid_gray", mid_gray)
    cv2.waitKey(0)

    # 将灰度图二值化 灰度值小于125的点置0，灰度值大于125的点置255
    ret, thresh = cv2.threshold(mid_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    # # 腐蚀2次 将图片中的一些毛刺或者很细小的东西给去掉 与之对应的还有膨胀 cv2.dilate() 都是针对二值化之后的图
    # thresh = cv2.erode(thresh, kernel, iterations=2)
    # cv2.imshow("erode", thresh)
    # cv2.waitKey(0)
    #
    # # 膨胀
    # thresh = cv2.dilate(thresh, kernel)
    # # thresh = cv2.Canny(thresh, 0, 60, apertureSize=3)
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)

    # 开运算， 先腐蚀后膨胀; 闭运算，先膨胀后腐蚀；梯度运算，膨胀-腐蚀；礼帽，原图-开运算结果；黑帽，闭运算-原图
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 查找轮廓 cv2.CHAIN_APPROX_SIMPLE 轮廓近似方式 它会移除所有冗余的点并压缩轮廓，结果只有四个点，从而节省内存。
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[1]  # 按面积排序获取目标轮廓，按实际的图来
    cir = cv2.arcLength(c, True)  # 获取闭合轮廓的周长
    c = cv2.approxPolyDP(c, cir / 100, True)  # 获取近似轮廓

    img_ = np.zeros(gray.shape, np.uint8)  # 生成黑色画布
    img_ = cv2.drawContours(img_, [c], 0, (255, 255, 255), 2)  # 画出轮廓

    cv2.imshow("contour", img_)
    cv2.waitKey(0)

    height = img_data.shape[0]
    weight = img_data.shape[1]
    four_points = np.array((

        (weight - 1, 0),
        (weight - 1, height - 1),
        (0, height - 1), (0, 0),),
        np.float32)
    target_points = np.array(c, np.float32)  # 统一格式
    m = cv2.getPerspectiveTransform(target_points, four_points)  # 获取透视变换矩阵
    res = cv2.warpPerspective(img_data, m, (weight, height))  # 透视变换，可保持直线不变形，但是平行线可能不再平行
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.imwrite("res.jpg", res)


if __name__ == '__main__':
    img = cv2.imread('./images/IMG_0221.jpg')
    image_correct1(img)
    image_correct(img)
