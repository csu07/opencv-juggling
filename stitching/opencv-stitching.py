import cv2.cv2 as cv2
import numpy as np

'''
1.按坐标截取图片
'''
'''
img = cv2.imread("Resources/cards.jpg")
#四张牌 4个坐标
# pts1 需要截取图片的四点坐标 pts2 结果图坐标 matrix 存储pts2的像素
width,height=250,350
#pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
pts1 = np.float32([[127,96],[248,97],[126,260],[247,260]])
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix=cv2.getPerspectiveTransform(pts1,pts2)
imgoutput=cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow("Image",img)
cv2.imshow("Output",imgoutput)
cv2.waitKey(0)
'''
'''
2.水平、竖直合并
img = cv2.imread('./img/A.jpg')

imgHor = np.hstack((img, img))
imgVer = np.vstack((img, img))

cv2.imshow("Horizontal", imgHor)
cv2.imshow("Vertical", imgVer)
cv2.waitKey(0)
'''


class Stitching(object):
    def __init__(self):
        self.ratio = 0.85
        self.min_match = 10
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.smoothing_window_size = 800

    def registration(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('./test/matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        return H

    def create_mask(self, img1, img2, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        H = self.registration(img1, img2)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1, img2, version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2
        result = panorama1 + panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result


if __name__ == '__main__':
    img1 = cv2.imread("/Users/a7/work/python/opencv-juggling/stitching/test/IMG_0284.jpg")
    img2 = cv2.imread("/Users/a7/work/python/opencv-juggling/stitching/test/IMG_0285.jpg")
    final = Stitching().blending(img1, img2)
    cv2.imwrite('./test/output.jpg', final)
