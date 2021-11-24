from cv2 import cv2
import numpy as np


def cut_img(img, num, overlap_factor):
    """img 是图像矩阵，num是切过后子图数目，因为统一为正方形切图，因此这里的num开方后需要为整数，
        overlap_factor 是切分后重叠部分的步长"""
    factor = int(np.sqrt(num))
    raw_shape = max(img.shape)
    cut_raw_shape = raw_shape // factor
    resize_shape = int(cut_raw_shape // 32) * 32  # 因为模型要求最小送入矩阵为32
    img = cv2.resize(img, (factor * resize_shape, factor * resize_shape))
    img_stacks = []  # 返回结果装载矩阵
    overlap_factor = overlap_factor
    cut_shape = int((factor * resize_shape + overlap_factor) / factor)  # 需要保证除以factor整除
    for i in range(factor):
        for ii in range(factor):
            img_temp = img[(ii * cut_shape - ii * overlap_factor):((ii + 1) * cut_shape - ii * overlap_factor),
                       (i * cut_shape - i * overlap_factor):((i + 1) * cut_shape - i * overlap_factor)]
            img_stacks.append(img_temp)

    return img_stacks


def calWeight(d, k):
    """
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    """

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def img_fusion(img1, img2, overlap, left_right=True):
    """
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    """
    # 这里先暂时考虑平行向融合
    w = calWeight(overlap, 0.05)  # k=5 这里是超参

    if left_right:  # 左右融合
        row1, col1 = img1.shape
        row2, col2 = img2.shape
        img_new = np.zeros((row1, col1 + col2 - overlap))
        img_new[0:row1, 0:col1] = img1
        w_expand = np.tile(w, (row1, 1))  # 权重扩增
        img_new[0:row1, (col1 - overlap):col1] = \
            (1 - w_expand) * img1[0:row1, (col1 - overlap):col1] + \
            w_expand * img2[0:row2, 0:overlap]
        img_new[:, col1:] = img2[:, overlap:]
    else:  # 上下融合
        row1, col1 = img1.shape
        row2, col2 = img2.shape
        img_new = np.zeros((row1 + row2 - overlap, col1))
        img_new[0:row1, 0:col1] = img1
        w = np.reshape(w, (overlap, 1))
        w_expand = np.tile(w, (1, col1))
        img_new[row1 - overlap:row1, 0:col1] = \
            (1 - w_expand) * img1[(row1 - overlap):row1, 0:col1] + \
            w_expand * img2[0:overlap, 0:col2]
        img_new[row1:, :] = img2[overlap:, :]
    return img_new


if __name__ == "__main__":
    import time

    start = time.time()
    img1 = cv2.imread("./img/a1.jpg", cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("./img/a2.jpg", cv2.IMREAD_UNCHANGED)
    img1 = cv2.resize(img1, (3094, 285))
    img2 = cv2.resize(img2, (3094, 285))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)  # 高斯模糊去噪点
    mid_gray1 = cv2.medianBlur(gray1, 3)  # 中值滤波忽略较高阶灰度和较低阶灰度

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)  # 高斯模糊去噪点
    mid_gray2 = cv2.medianBlur(gray2, 3)  # 中值滤波忽略较高阶灰度和较低阶灰度

    gray1 = (gray1 - gray1.min()) / gray1.ptp()
    gray2 = (gray2 - gray2.min()) / gray2.ptp()
    img_new = img_fusion(gray1, gray2, overlap=128, left_right=True)
    img_new = np.uint8(img_new * 255)
    cv2.imwrite('./img/res.jpg', img_new)
    end = time.time()
    print(end - start)
