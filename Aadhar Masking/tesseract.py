import numpy as np
import cv2
import os
import math
from PIL import Image
import pytesseract
import codecs
import pyocr
import pyocr.builders



def resized(frame, w):
    ht, wt = frame.shape
    return (w,(int)(ht/(wt/w)))

for image_name in os.listdir("TestFront"):

    print("-------------------------Start----------------------------------------------")
    img = cv2.imread("TestFront/"+image_name, 0)
    # img = cv2.imread("TestFront/test2.jpg", 0)

    image = cv2.resize(img.copy(), (resized(img, 600)))
    h, w = image.shape

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))




    cv2.imshow("TopHat", tophat)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    blackHat = cv2.morphologyEx(tophatB, cv2.MORPH_BLACKHAT, rectKernel)
    cv2.imshow("blackHat", blackHat)

    ret3,th3 = cv2.threshold(blackHat,1,255,cv2.THRESH_BINARY)
    cv2.imshow("Otsu", th3)

    edge = cv2.Canny(tophatB, 0, 10)
    cv2.imshow("Canny", edge)

    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, sqKernel)
    cv2.imshow("ImageCloses", edges)

    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    blackhat = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, sqKernel)
    cv2.imshow("ImageClose", blackhat)

    filename = "{}.png".format(os.getpid())
    print(filename)
    cv2.imwrite(filename, blackHat)
    text = pytesseract.image_to_string(Image.open(filename))
    print (text)

    tool = pyocr.get_available_tools()[0]
    builder = pyocr.builders.TextBuilder()

    print("-----------------------------------------------------------------------")
    langs = tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    lang = langs[0]

    txt = tool.image_to_string(Image.open(filename),lang=lang,builder=builder)
    print(txt)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
