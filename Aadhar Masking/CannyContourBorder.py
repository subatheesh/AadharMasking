import numpy as np
import cv2
import os
import math


def resized(frame, w):
    ht, wt = frame.shape
    return (w,(int)(ht/(wt/w)))



for image_name in os.listdir("TestFront"):

    img = cv2.imread("TestFront/"+image_name, 0)
    # img = cv2.imread("TestFront/test2.jpg", 0)

    image = cv2.resize(img.copy(), (resized(img, 600)))
    h, w = image.shape

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

    tophatB  = cv2.GaussianBlur(image,(5,5),0)

    _,th = cv2.threshold(tophatB, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("th", th)
    edges = cv2.Canny(th, 10, 50)
    cv2.imshow("Canny", edges)

    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges,sqKernel, iterations=1)
    cv2.imshow("ImageCloses", edges)

    _, contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x,y,w,h = cv2.boundingRect(approx)
        area = cv2.contourArea(approx)
        if(len(approx) >= 4 and area > 1000):
            # image =  cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            image = cv2.drawContours(image, [approx], 0,(0,255,0), 2)



    cv2.imshow("Image",image)
    # cv2.imshow("contours2", imge)
    # cv2.imshow("closing", closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
