import cv2
import numpy as np
import os

count = 0
for image in os.listdir("Img"):
    count += 1
    # if(count < 6):
    #     continue
    if(count > 10):
        break
    img = cv2.imread('Img/'+image, 0)
    width = img.shape[1]
    height = img.shape[0]

    if(width>height):
        w = 600
        h = (int)(height/(width/w))
        img = cv2.resize(img, (w,h))
    else:
        h = 600
        w = (int)(width/(height/h))
        img = cv2.resize(img, (w,h))

    edges = cv2.Canny(img, 150, 250)

    cv2.imshow("Edges", edges)
    img = cv2.GaussianBlur(img,(5,5),0)


    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    tophat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKernel)

    cv2.imshow("Black", tophat)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
    	ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    cv2.imshow("Grad", gradX)

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    cv2.imshow("FirstClose",gradX)
    thresh = cv2.threshold(gradX, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("OTSU", thresh)
    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    cv2.imshow("SecondClose", thresh)
    cv2.imshow("TopHat", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
