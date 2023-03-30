import cv2
import numpy as np


def detect_text_lines(img, min_area=500, max_distance=50):
    #####有两个阈值min_area，max_distance，字面意思
    # 初始化矩形列表
    rects = []
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 应用二值化阈值
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    # 创建矩形结构元素
    rec = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    # 腐蚀二值图像
    dilate0 = cv2.erode(binary, rec)
    # 反转腐蚀后的图像
    erode2 = cv2.bitwise_not(dilate0)
    # 查找轮廓
    counts, _ = cv2.findContours(erode2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历每个轮廓
    for i in range(len(counts)):
        # 如果轮廓面积小于最小面积，则忽略该轮廓
        if cv2.contourArea(counts[i]) < min_area:
            continue
        # 计算轮廓的边界矩形
        rect1 = cv2.boundingRect(counts[i])
        # 将边界矩形添加到矩形列表中
        rects.append(rect1)

    # 删除包含在较大文本框中的较小文本框
    final_rects = []
    for rect1 in rects:
        x1, y1, w1, h1 = rect1
        is_contained = False
        for rect2 in rects:
            x2, y2, w2, h2 = rect2
            if x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
                is_contained = True
                break
        if not is_contained:
            final_rects.append(rect1)

    # 合并附近的文本框
    merged_rects = []
    for rect1 in final_rects:
        x1, y1, w1, h1 = rect1
        is_merged = False
        for i in range(len(merged_rects)):
            rect2 = merged_rects[i]
            x2, y2, w2, h2 = rect2
            if abs(y1 - y2) <= max_distance:
                merged_rects[i] = (
                    min(x1, x2), min(y1, y2), max(x1 + w1, x2 + w2) - min(x1, x2), max(y1 + h1, y2 + h2) - min(y1, y2))
                is_merged = True
                break
        if not is_merged:
            merged_rects.append(rect1)

    return merged_rects


def line_detect(img):
    # Detect text lines
    rects = detect_text_lines(img)

    # Draw rectangles around detected text lines
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


# img = cv2.imread("./example/1.png")
# img = line_detect(img)
# cv2.imwrite('./temp/1er.jpg', img)
