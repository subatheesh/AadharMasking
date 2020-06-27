import cv2 as cv
import math
import argparse
import os
import io
import numpy as np
import math
from PIL import Image
import pytesseract
import codecs
import pyocr
import pyocr.builders
from google.cloud import vision


def resizedH(frame, h):
    ht, wt, ch = frame.shape
    return ((int)(wt/(ht/h)), h)

def checkNumbers(text):
    NumCount = 0
    IsNum = False
    IsNoChar = True
    for x in text:
        if(ord(x)==10 or ord(x)==32):
            pass
        elif (ord(x)>47 and ord(x)<58):
            IsNum = True
            NumCount += 1
        else:
            IsNoChar = False
    return (IsNum and IsNoChar, NumCount)

def GetAadhaar(des):
    Aadhaar = []
    LineDes = []
    lines = des.split('\n')
    for i,line in enumerate(lines):
        status, count = checkNumbers(line)
        LineDes.append([status, count, line])
        if(status):
            if(count == 12):
                Aadhaar.append(line)
    if(len(Aadhaar) >= 1):
        return Aadhaar[0].split(" ")
    else:
        count = 0
        newLine = []
        print
        for k in LineDes:
            merge = False
            if(count>0):
                if(LineDes[count-1][0] and LineDes[count][0] and LineDes[count-1][1]%4 == 0 and LineDes[count][1]%4 == 0 and LineDes[count-1][1]+ LineDes[count][1] <= 12):
                    merge = True
                    newLine[-1] = [True, LineDes[count-1][1]+ LineDes[count][1], LineDes[count-1][2] + " " + LineDes[count][2]]
            if(not merge):
                count += 1
                newLine.append(k)
            if(newLine[-1][0] and newLine[-1][1] == 12):
                return newLine[-1][2].split(" ")
    return False

def CloudVisionOCR(img):
    file_name = "GC.jpg"
    cv.imwrite(file_name, img)

    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print(len(texts))
    if(len(texts) > 0):
        print(texts[0].description)
        aadhar = GetAadhaar(texts[0].description)
        print(aadhar)
        if(type(aadhar) == bool):
            print("Reject")
            return False
        else:
            aadhar_len = len(aadhar)
            index = 0
            for i,text in enumerate(texts):
                print('\n"{}"'.format(text.description), index, i)
                if(text.description == aadhar[index]):
                    index += 1
                else:
                    index = 0
                if(index == aadhar_len):
                    totalchar = 8
                    for k in range(aadhar_len):
                        ratio = 1.0
                        cur = texts[i+k-aadhar_len+1]
                        prevTotal = totalchar
                        print(prevTotal)
                        if(totalchar < len(cur.description)):
                            ratio = float(totalchar)/float(len(cur.description))
                            totalchar = 0
                        else:
                            totalchar -= len(cur.description)
                        if(prevTotal > 0):
                            vertices = ([[vertex.x, vertex.y]for vertex in cur.bounding_poly.vertices])
                            print(ratio)
                            print(vertices)
                            if(ratio < 1.0):
                                vertices[1] = [vertices[0][0] + ratio*(vertices[1][0]-vertices[0][0]), vertices[0][1] + ratio*(vertices[1][1]-vertices[0][1])]
                                vertices[2] = [vertices[3][0] + ratio*(vertices[2][0]-vertices[3][0]), vertices[3][1] + ratio*(vertices[2][1]-vertices[3][1])]
                                print(vertices)
                            mask = np.array( [vertices], dtype=np.int32 )
                            cv.fillPoly( img, mask, 0 )
                    index = 0
            # cv.imshow("ds", cv.resize(img, (resizedH(img, 600))))
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            return True

c = 0
client = vision.ImageAnnotatorClient()
# CloudVisionOCR(cv.imread("Test/c41d10d9-f42b-4339-912d-43cb8ea3c63c.jpg", 1))
dir = "Test"
for image_name in os.listdir(dir):
    c += 1
    print(c,image_name)
    frame = cv.imread(dir+"/"+image_name, 1)
    # CloudVisionOCR(frame)
    if(CloudVisionOCR(frame)):
        cv.imwrite("CloudVisionResult/Success/"+image_name, frame)
    else:
        cv.imwrite("CloudVisionResult/Failure/"+image_name, frame)
