import cv2
import numpy as np


def detect_text_lines(img, min_area=500, max_distance=60):
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

    merged_rects = []
    for rect1 in rects:
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
    
    finals = []
    for rect1 in merged_rects:
        x1, y1, w1, h1 = rect1
        is_contained = False
        for rect2 in merged_rects:
            x2, y2, w2, h2 = rect2
            if x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
                is_contained = True
                break
        if not is_contained:
            finals.append(rect1)
    # 对矩形按中心坐标排序
    sorted_rects = sorted(finals, key=lambda r: (r[1] + r[3] / 2, r[0] + r[2] / 2))

    # 将矩形转换为二维数组
    rows = []
    current_row = []
    for rect in sorted_rects:
        if not current_row or rect[1] + rect[3] / 2 > current_row[-1][1] + current_row[-1][3] / 2:
            # 新的一行
            if current_row:
                rows.append(current_row)
            current_row = [rect]
        else:
            # 同一行
            current_row.append(rect)
    if current_row:
        rows.append(current_row)
    return rows


def darw_rect(img):
    # Detect text lines
    rects = detect_text_lines(img)

    # Draw rectangles around detected text lines
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 返回的是被标注了框的原图
    return img
def new_imgs(img):
    new_imgs = []
    rows = detect_text_lines(img)
    for row in rows:
        for rect in row:
            x, y, w, h = rect

            # 裁剪矩形部分
            roi = img[y:y+h, x:x+w]

            # 将裁剪出的图像添加到新图像列表中
            new_imgs.append(roi)
    return new_imgs
    
if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\Administrator\Documents\GitHub\UOCR\example\example (2).png")
    # img = drat_rect(img)
    # cv2.imwrite('./temp/1er.jpg', img)
    imgs = new_imgs(img)
    i = 0
    for img in imgs:
        cv2.imwrite(rf'C:\Users\Administrator\Documents\GitHub\UOCR\temp\new{i}.png',img)
        i += 1