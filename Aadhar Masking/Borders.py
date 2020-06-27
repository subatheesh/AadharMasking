import numpy as np
import cv2
import os
import math
from PIL import Image
import pytesseract


def resized(frame, w):
    ht, wt = frame.shape
    return (w,(int)(ht/(wt/w)))

def HoughLines(imageb, name, img):
    lines = cv2.HoughLines(imageb,1,np.pi/180,255)
    # print (len(lines))
    count = 0
    for line in lines:
        # if(count > 10):
        #     break
        # count +=1
        for rho,theta in line:
            # print(rho, theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow(name, img)

def HoughLinesP(imageb, name, img):
    # print(sum(imageb(:) == 0 ))
    # print(sum(imageb(:) == 1 ))
    minLineLength = 500
    maxLineGap = 100
    lines = cv2.HoughLinesP(imageb,1,np.pi/180,10,minLineLength,maxLineGap)
    if lines is not None:
        # print(len(lines))
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow(name, img)


for image_name in os.listdir("TestFront"):

    img = cv2.imread("TestFront/"+image_name, 0)
    # img = cv2.imread("TestFront/test2.jpg", 0)

    image = cv2.resize(img.copy(), (resized(img, 600)))
    h, w = image.shape

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

    tophatB  = cv2.GaussianBlur(image,(5,5),0)
    tophat = cv2.adaptiveThreshold(tophatB,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV,13,2)


    cv2.imshow("TopHat", tophat)

    edges = cv2.Canny(tophatB, 10, 50)
    cv2.imshow("Canny", edges)

    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges,sqKernel, iterations=1)
    cv2.imshow("ImageCloses", edges)





    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    blackhat = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, sqKernel)
    cv2.imshow("ImageClose", blackhat)

    # filename = "{}.png".format(os.getpid())
    # print(filename)
    # cv2.imwrite(filename, tophatB)
    # text = "dfadsf"
    # text = pytesseract.image_to_string(Image.open(filename))
    # print (text)


    # HoughLines(blackhat,"HoughLinesG", image.copy())
    HoughLinesP(blackhat,"HoughLinesPG", image.copy())

    # HoughLines(edges,"HoughLinesCG", image.copy())
    HoughLinesP(edges,"HoughLinesPCG", image.copy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
