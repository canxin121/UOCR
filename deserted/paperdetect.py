import numpy as np
import cv2
from line_crop import line_detect
from pathlib import Path
def remove_duplicate_contours(contours, threshold):
    # 计算每个轮廓的中心点
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
        else:
            centers.append((0, 0))

    # 检查并删除重复的轮廓
    new_contours = []
    for i in range(len(contours)):
        duplicate = False
        for j in range(i + 1, len(contours)):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            if dist < threshold:
                duplicate = True
                break
        if not duplicate:
            new_contours.append(contours[i])

    return new_contours


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


def paper_detect(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()

    image = resize(orig, height=500)

    # 转灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯滤波，去除噪音点
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    edged = cv2.Canny(gray, 75, 200)

    # print('get edged')
    cv2.imwrite("./temp/edge.png", edged)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    cnts = remove_duplicate_contours(cnts, threshold=5)
    screencnt = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)

        if len(approx) == 4:
            screencnt.append(approx)
    # 框起来调试
    i = 0

    for s in screencnt:
        contourpng = cv2.drawContours(image,[s],-1,(0,255,0),2)
        cv2.imwrite(f"./temp/contour{i}.png",contourpng)
        i += 1
    i = 0

    ##############
    warpeds = []
    for screen in screencnt:
        # cv2.imwrite("./temp/why.png",orig)
        warped = four_point_transform(orig, screen.reshape(4, 2) * ratio)
        warpeds.append(warped)
        # 把ref写入scan.jpg
        cv2.imwrite(f'./temp/scan{i}.jpg', warped)
        i+=1
    return warpeds
root = str(Path.cwd())
cn = 0
image=cv2.imread(root + '/example/test2.png')
print(1)
for i in paper_detect(image):
    newimg = line_detect(i)
    cv2.imwrite(f"{root}/temp/paper{cn}.png",newimg)
    cn += 1

