import numpy as np
import cv2
import os
import math
from PIL import Image
import pytesseract

def resized(frame, w):
    ht, wt = frame.shape
    return (w,(int)(ht/(wt/w)))



selected = -1
points = [(10, 10), (490, 10), (490, 304), (10, 304)]

def findNearCircle(x,y):
    min_index = -1
    min_val = math.inf
    for i,point in enumerate(points):
        dist = math.sqrt((x - point[0])**2 + (y - point[1])**2)
        if dist < min_val:
            min_val = dist
            min_index = i
    return min_index;


def mouseCallback(event, x, y, flags, params):
    global selected
    if event == cv2.EVENT_LBUTTONDOWN:
        selected = findNearCircle(x,y)
        points[selected] = (x,y)

    if event == cv2.EVENT_LBUTTONUP:
        selected = -1

    if event == cv2.EVENT_MOUSEMOVE:
        if selected >= 0:
            points[selected] = (x,y)



for image_name in os.listdir("TestFront"):
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouseCallback)

    img = cv2.imread("TestFront/"+image_name, 0)
    # img = cv2.imread("TestFront/test2.jpg", 0)

    image = cv2.resize(img.copy(), (resized(img, 1000)))
    h, w = image.shape
    points = [(10, 10), (w-10, 10), (w-10, h-10), (10, h-10)]

    frame = image.copy()
    while True:
        img = image.copy()
        for i, point in enumerate(points):
            cv2.circle(img, point, 5, (0,0,255), -1)
            cv2.line(img, point, points[(i+1)%4], (0,0,255), 2)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == 32 or key == 13:
            break
    cv2.destroyWindow("Image")
    # points = [(84, 36), (441, 54), (450, 359), (40, 354)]
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[1000,0],[1000,628],[0,628]])
    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(frame,M,(1000,628))
    cv2.imshow("Images", dst)
    cv2.waitKey(0)



    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 6))
    rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 12))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    rectKernalinv = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

    tophat = cv2.morphologyEx(dst, cv2.MORPH_BLACKHAT, rectKernel)
    cv2.imshow("Images", tophat)
    cv2.waitKey(0)

    # filename = "{}.png".format(os.getpid())
    # print(filename)
    # cv2.imwrite(filename, dst)
    # text = "dfadsf"
    # text = pytesseract.image_to_string(Image.open(filename))
    # print (text)
    #
    # cv2.waitKey(0);

    gradX = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, rectKernel)
    cv2.imshow("Images", gradX)
    cv2.waitKey(0)

    thresh = cv2.threshold(gradX, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Images", thresh)
    cv2.waitKey(0)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel2)
    cv2.imshow("Images", thresh)
    cv2.waitKey(0)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernalinv)
    cv2.imshow("Images", thresh)
    cv2.waitKey(0)

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        dst = cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # dst = cv2.drawContours(dst,[box],0,(0,0,255),2)

    cv2.imshow("Images", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
