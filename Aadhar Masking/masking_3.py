# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:24:26 2019

@author: abhishekkaleidofin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:02:46 2019

@author: abhishekkaleidofin

Masked Images stored in Result3

Total Images 201

Correct Detections = 169

Accuracy of 84%

"""

import numpy as np
import imutils
import cv2
# from pyimagesearch.transform import four_point_transform
import os
# import inspect
from matplotlib import pyplot as plt
import math
# from skimage.filters import threshold_local
# import timeit

# directory = 'C:\\Users\\abhishekkaleidofin\\Desktop\\Work\\Image Processing\\Aadhar Masking\\new_aadhaar_images'
# destDir = 'C:\\Users\\abhishekkaleidofin\\Desktop\\Work\\Image Processing\\Aadhar Masking\\new_aadhaar_images\\Result3'

directory = 'C:\\Users\\subatheeshkaleidofin\\Aadhar Masking\\TestFront'
destDir = 'C:\\Users\\subatheeshkaleidofin\\Aadhar Masking\\TestFront\\Result3'

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def scan(image):
    image = cv2.imread(image)
    ratio = image.shape[0] / 500.0
    orig_1 = image.copy()

    image_1 = imutils.resize(image, height = 500)

    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    paper = cv2.medianBlur(image_1,13)
    #paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    #thresh_gray = cv2.adaptiveThreshold(paper,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    # ret, thresh_gray = cv2.threshold(cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY), 150, 200, cv2.THRESH_BINARY)
    ret, thresh_gray = cv2.threshold(paper,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh_gray = cv2.adaptiveThreshold(paper,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    cv2.imshow('paper', thresh_gray)
    cv2.waitKey(0)
    _, contours, hier = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    """
    count = 1
    for c in contours:
        if count>2:
            break
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a green 'nghien' rectangle
        temp_img = cv2.drawContours(image_1.copy(), [box], 0, (0, 255, 0),1)
        cv2.imshow('contour_{}'.format(count), temp_img)
        cv2.waitKey(0)
        count+=1



    max_area = 0
    for c in contours:
        print(cv2.contourArea(c))
        if (max_area < cv2.contourArea(c)):
            cnt = c
    """

    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a green 'nghien' rectangle
    cv2.drawContours(image_1, [box], 0, (0, 255, 0),1)

    """
    r = cv2.selectROI(image_2, False)

    imCrop = image_2[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    """
    '''
    cv2.imshow('final_contour', image_1)
    cv2.waitKey(0)
    '''

    warped = four_point_transform(orig_1, box.reshape(4, 2) * ratio)
    print(warped.shape)
    cv2.imwrite('four_pt_img_1.jpg',warped)


    cv2.imshow('Orginal',imutils.resize(orig_1, height = 650))

    cv2.imshow('paper',imutils.resize(warped, height = 650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def masker(image):

    scan(image)

    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,3))   #(9,6)
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # load the input image, resize it, and convert it to grayscale
    image_new = cv2.imread('four_pt_img_1.jpg')
    image_new = imutils.resize(image_new, height = 400)
    #gray = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)
    #width = int(image_new.shape[1]-image_new.shape[1]*0.25)
    crpImg = image_new[270:,:]
    gray = cv2.cvtColor(crpImg, cv2.COLOR_BGR2GRAY)

    #ret, thresh_gray = cv2.threshold(gray, 130, 200, cv2.THRESH_BINARY_INV)

    '''
    cv2.imwrite('cropped.jpg',gray)

    cv2.imshow('blah', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    '''
    cv2.imshow('blah',tophat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    '''
    cv2.imshow('blah',gradX)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    '''
    cv2.imshow('blah',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    locs = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # since credit cards used a fixed size fonts with 4 groups
        # of 4 digits, we can prune potential contours based on the
        # aspect ratio
        if ar > 1.0 and ar < 5.0:
            # contours can further be pruned on minimum/maximum width
            # and height
            if (w > 10 and w < 100) and (h > 5 and h < 50):
                # append the bounding box region of the digits group
                # to our locations list
                locs.append((x, y, w, h))

    # Sort contours by y-axis
    locs = sorted(locs, key=lambda x:x[1])

    '''
    # loop over the 4 groupings of 4 digits
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []

        # extract the group ROI of 4 digits from the grayscale image,
        # then apply thresholding to segment the digits from the
        # background of the credit card
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # detect the contours of each individual digit in the group,
        # then sort the digit contours from left to right
        try:
            digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            digitCnts = imutils.grab_contours(digitCnts)
            digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
        except Exception as e:
            print(e)
            continue
    '''

    # Get contours of the first 8 aadhar digits
    if len(locs)>1:
        num_locs = [locs[0]]
        min_y = locs[0][1]
        for x in locs[1:]:
            if x[1]<min_y+25:
                num_locs.append(x)

        final_locs = sorted(num_locs, key=lambda x:x[0])[:2]

        if len(final_locs)==2:
            # Draw rect over first 8 aadhar digits
            cv2.rectangle(image_new, (final_locs[0][0],final_locs[0][1]+270),
                                 (final_locs[1][0]+final_locs[1][2],final_locs[1][1]+final_locs[1][3]+270),
                                 (255,0,0), 2)

    '''
    for i in range(len(final_locs)):
        c = locs[0]
        cv2.rectangle(image_new, (c[0], c[1]+270),  (c[0]+c[2],c[1]+c[3]+270), (255,0,0), 2)
        cv2.imshow('masked',image_new)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''

    '''
    cv2.imshow('masked',image_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return image_new

    #cv2.imwrite('time.jpg',image_new)

def master_masker(directory, destDir):
    c = 1
    for filename in os.listdir(directory):
        if filename.endswith('jpg'):
            print(filename)
            maskedImg = masker(directory+'\\'+filename)
            try:
                cv2.imwrite(destDir+'\\'+str(c)+'.jpg',maskedImg)
                c+=1
            except Exception as e:
                print(e)
                continue


master_masker(directory, destDir);
