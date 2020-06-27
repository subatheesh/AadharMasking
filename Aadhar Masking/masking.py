import numpy as np
import cv2
import os

count = 0
for image in os.listdir("Img"):
    count += 1
    if(count < 15):
        pass
    if(count > 20):
        break
    img = cv2.imread('Img/'+image, 0)
    img = cv2.resize(img,(500,500))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img,(5,5),0)
    cv2.imshow("Or",img);
    mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    cv2.imshow("mask", mask)
    kernel = np.ones((2,2),np.uint8)
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel, iterations = 1)
    opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel, iterations = 1)
    # edge = cv2.Canny(mask, 100, 200)


    opening = cv2.bitwise_not(mask)

    _, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[4]
    # epsilon = 0.001 * cv2.arcLength(contours, True)
    # approx = cv2.approxPolyDP(contours, epsilon, True)

    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x,y,w,h = cv2.boundingRect(approx)
        area = cv2.contourArea(cnt)
        # print(area)
        if(area > 50):
            img = cv2.rectangle(img,(x,y),(w+x,y+h),(0,255,0),3)
    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # im = cv2.drawContours(img,[box],0,(0,0,255),2)
    # imge = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    # image = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    # x,y,w,h = cv2.boundingRect(cnt)
    # img2 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow("contours", image)
    # cv2.imshow("contour3",im)
    cv2.imshow(image,img)
    # cv2.imshow("contours2", imge)
    # cv2.imshow("closing", closing)
    cv2.imshow("opening", opening)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('Test/test2.jpg',1)
# img = cv2.resize(img, (600,400))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img2 = gray.copy()
# template = cv2.imread('Template/template.jpg',0)
# template = cv2.resize(template, (600,400))
# w, h = template.shape[::-1]
# cv2.imshow("test",template);
# # All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#
# for meth in methods:
#     img3 = img2.copy()
#     method = eval(meth)
#
#     # Apply template Matching
#     res = cv2.matchTemplate(img3,template,method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#
#     cv2.rectangle(img,top_left, bottom_right, 128, 10)
#
#     cv2.imshow(meth, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    #
    # plt.show()



# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# MIN_MATCH_COUNT = 10
#
# img1 = cv2.imread('Template/template.jpg',0)          # queryImage
# img2 = cv2.imread('Test/test3.jpg',0) # trainImage
# # img2 = cv2.GaussianBlur(img2,(5,5),0)
# # Initiate SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
#
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(des1,des2,k=2)
#
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)
#
# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#
#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#     img3 = img2.copy()
#     img3 = cv2.resize(img2, (1000,600))
#     cv2.imshow("test", img3)
# else:
#     print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
#     matchesMask = None
#
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
#
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#
# plt.imshow(img3, 'gray'),plt.show()
