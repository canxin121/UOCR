import cv2
from line_crop import new_imgs
import numpy as np
#此函数用于计算rect中间点的xy坐标
def contour_center(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x + w / 2, y + h / 2


def crop_digit(img):
    imgs = new_imgs(img)
    j = 0
    digits_ = []
    for img in imgs:
        h_min, w_min = img.shape[:2]
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 对图像进行二值化处理
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        # 为形态学操作定义一个内核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        # 执行形态学闭合
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # 查找图像中的轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        # 对轮廓按中心坐标的x从小到大排序
        contours = sorted(contours, key=contour_center)
        # 遍历每个轮廓
        row = []
        for i, contour in enumerate(contours):
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            # 裁剪出单个字符的图像
            roi = img[y:y + h, x:x + w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 对图像进行二值化处理
            _, thresh = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            # 获取图像的当前形状
            h, w = thresh.shape
            if max(h,w) > h_min*0.3 and (w < h *1.5 or h*w > h_min * w_min *0.05):
                # 计算需要填充多少行和列才能使其成为正方形
                if h > w:
                    pad_size = (h - w) // 2
                    padded = np.pad(thresh, ((15, 15), (pad_size+15, pad_size+15)), 'constant', constant_values=0)
                else:
                    pad_size = (w - h) // 2
                    padded = np.pad(thresh, ((pad_size+15, pad_size+15), (15, 15)), 'constant', constant_values=0)
                # 为形态学操作定义一个内核
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                # 执行形态学闭合
                closed_ = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
                newimg = cv2.resize(closed_, (28, 28), interpolation=cv2.INTER_NEAREST)
                row.append(newimg)
        digits_.append(row)
    return digits_
