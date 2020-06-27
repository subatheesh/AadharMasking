# Import required modules
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

client = vision.ImageAnnotatorClient()

def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    return [detections, confidences]

def checkDirection(indices, boxes):
    count = 0
    print(len(boxes))
    for i in indices:
        if(boxes[i[0]][1][0]>boxes[i[0]][1][1]):
            count += 1
        else:
            count -= 1
    print (count)
    return count>=0

def isMergable(newboxes, box, index):
    if(len(newboxes) > 0):
        for k in newboxes[-1]:
            if(abs(k[0][index] - box[0][index]) < 10 ):
                return True
    return False

def mergeBoxes(boxes, indices):
    newboxes = []
    index = 0 if checkDirection(indices, boxes) else 1
    for i in indices:
        if(isMergable(newboxes, boxes[i]), index):
            newboxes[-1].append(boxes[i])
        else:
            newboxes.append([boxes[i]])
    return newboxes

def sortIndices(indices, boxes):
    index = 1 if checkDirection(indices, boxes) else 0
    print (index)
    def customSort(x):
        return boxes[x[0]][0][index]
    return sorted(indices, key = customSort)


def resized(frame, w):
    ht, wt, ch = frame.shape
    return (w,(int)(ht/(wt/w)))

def resizedH(frame, h):
    ht, wt, ch = frame.shape
    return ((int)(wt/(ht/h)), h)

def cropImage(img, box, width, height):
    box = np.int0(box)
    src_pts = box.astype("float32")

    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv.warpPerspective(img, M, (width, height))
    return warped

def TesseractText(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Gblur  = cv.GaussianBlur(gray,(7,7),0)
    # th = cv.adaptiveThreshold(Gblur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv.THRESH_BINARY_INV,11,2)
    # sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    # im = cv.morphologyEx(th, cv.MORPH_CLOSE, sqKernel)
    cv.imshow("pro",gray)
    filename = "p.png".format(os.getpid())
    # print(filename)
    cv.imwrite(filename, gray)
    return pytesseract.image_to_string(Image.open(filename))

def TesseractTextUnprocessed(img):
    filename = "up.png".format(os.getpid())
    # print(filename)
    cv.imwrite(filename, img)
    return pytesseract.image_to_string(Image.open(filename))

def CloudVisionOCR(img):
    file_name = "GC.jpg"
    cv.imwrite(file_name, img)

    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:', texts[0].description if len(texts)>0 else "")

    return texts[0].description if len(texts)>0 else "NULL"
    # for text in texts:
    #     print('\n"{}"'.format(text.description))
    #
    #     vertices = (['({},{})'.format(vertex.x, vertex.y)
    #                 for vertex in text.bounding_poly.vertices])
    #
    #     print('bounds: {}'.format(','.join(vertices)))

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

if __name__ == "__main__":
    confThreshold = 0.5
    nmsThreshold = 0.4
    inpWidth = 640
    inpHeight = 640
    model = "frozen_east_text_detection.pb"

    net = cv.dnn.readNet(model)

    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    dir = "Test"
    for image_name in os.listdir(dir):
        print(image_name)
        frame = cv.imread(dir+"/"+image_name, 1)

        orginal = frame.copy()

        frame = cv.resize(frame, (resizedH(frame, 600)))

        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        net.setInput(blob)
        output = net.forward(outputLayers)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = decode(scores, geometry, confThreshold)

        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        indices = sortIndices(indices,boxes)

        filteredIndices = []

        for i in indices:
            newbox = (boxes[i[0]][0],(boxes[i[0]][1][0] + 10, boxes[i[0]][1][1] + 10), boxes[i[0]][2])
            vertices = cv.boxPoints(newbox)

            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH

            cropImg = cropImage(frame.copy(), vertices, (int) (newbox[1][0]*rW), (int)(newbox[1][1]*rH));

            text = TesseractText(cropImg)
            textu = TesseractTextUnprocessed(cropImg)
            Gtext = CloudVisionOCR(cropImg)

            flag = checkNumbers(Gtext)
            print(flag)
            if flag[0]:
                filteredIndices.append((i[0], Gtext, flag[1]))

            # print("")
            # print (text, vertices[1])
            # print (textu,  "Unprocessed")
            # cv.imshow("cropImage", cropImg)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # for j in range(4):
            #     p1 = (vertices[j][0], vertices[j][1])
            #     p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            #     # if(text != "" or textu != ""):
            #         # cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA)
            #     # else:
            #     cv.line(frame, p1, p2, (255, 0, 0), 2, cv.LINE_AA)
            #     # cv.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
            # cv.imshow("fdas",frame)


        # Put efficiency information
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        print(filteredIndices)

        newboxes =  mergeBoxes(boxes, filteredIndices)
        print (newboxes)

        for i in filteredIndices:
            newbox = (boxes[i[0]][0],(boxes[i[0]][1][0] + 10, boxes[i[0]][1][1] + 10), boxes[i[0]][2])
            vertices = cv.boxPoints(newbox)

            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH

            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                # if(text != "" or textu != ""):
                    # cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA)
                # else:
                cv.line(frame, p1, p2, (255, 0, 0), 2, cv.LINE_AA)

        # print(frame.shape)
        # Display the frame
        # image = cv.resize(frame.copy(), (resized(frame, 500)))
        # cv.imshow("kWinName",frame)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        cv.imwrite("ResultTest/out-{}".format(image_name),frame)
