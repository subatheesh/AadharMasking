import numpy as np
import cv2
import os
import math


def resized(frame, w):
    ht, wt = frame.shape
    return (w,(int)(ht/(wt/w)))


img = cv2.imread("TestFront/test.jpg", 0)
img = cv2.resize(img.copy(), (resized(img, 500)))
bi = cv2.bilateralFilter(img, 5, 75, 75)
dst = cv2.cornerHarris(bi, 2, 3, 0.1)

cv2.imshow('img', img)

corners = cv2.goodFeaturesToTrack(bi, 150, 0.1, 20)
corners = np.int0(corners)
images = img.copy()
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(images, (x, y), 5, (0, 0, 255), -1)
cv2.imshow("images", images)


mask = np.zeros_like(img)


mask[dst>0.01*dst.max()] = 255
cv2.imshow('mask', mask)
cv2.imshow('bi',bi)

# img[dst > 0.01 * dst.max()] = [0, 0, 255]   #--- [0, 0, 255] --> Red ---
# cv2.imshow('dst', img)

coor = np.argwhere(mask)
coor_list = [l.tolist() for l in list(coor)]
coor_tuples = [tuple(l) for l in coor_list]

thresh = 20

def distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

coor_tuples_copy = coor_tuples

i = 1
for pt1 in coor_tuples:

    # print(' I :', i)
    for pt2 in coor_tuples[i::1]:
        # print(pt1, pt2)
        # print('Distance :', distance(pt1, pt2))
        if(distance(pt1, pt2) < thresh):
            coor_tuples_copy.remove(pt2)
    i+=1

img2 = img.copy()
for pt in coor_tuples:
    cv2.circle(img2, tuple(reversed(pt)), 3, (0, 0, 255), -1)
cv2.imshow('Image with 4 corners', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
