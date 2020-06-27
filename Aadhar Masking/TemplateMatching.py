import numpy as np
import cv2
import os
import queue
import heapq


def resized(frame, w):
    ht, wt = frame.shape
    return (w,(int)(ht/(wt/w)))


def matchTemplate(template, image):
    w1, h1 = template.shape[::-1]
    res = cv2.matchTemplate(template, image, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w1, top_left[1] + h1)

    cv2.rectangle(image,top_left, bottom_right, 255, 5)
    top_left = max_loc
    bottom_right = (top_left[0] + w1, top_left[1] + h1)

    cv2.rectangle(image,top_left, bottom_right, 255, 2)
    # cv2.imshow("Orginal_T1",template)
    # cv2.imshow("Orginal1", image)
    # cv2.waitKey(0)
    return top_left, max_val


for image_name in os.listdir("TestFront"):
    intervals = [50, 20, 10, 5, 2, 1]
    min_range = 300
    max_range = 900

    img = cv2.imread("Template/templateHead.jpg", 0)
    img2 = cv2.imread("TestFront/"+image_name, 0)
    # img2 = cv2.imread("TestFront/test0.jpg", 0)
    img = cv2.resize(img, (300,40))
    max_correlation = 0
    top_left_p = (0,0)
    width_p = 0
    w , h = img.shape[::-1]
    for interval in intervals:
        q = queue.Queue(maxsize = 3)
        prev_corr = 0
        for testWidth in range(min_range, max_range + interval, interval):
            image = cv2.resize(img2.copy(), (resized(img2, testWidth)))

            topleft, correlation = matchTemplate(img, image)

            if(q.full()):
                q.get()
            if(correlation < prev_corr):
                while (not q.empty()):
                    _, min_range = q.get()
                max_range = testWidth
                break
            q.put((correlation,testWidth))
            prev_corr = correlation
            max_correlation = correlation
            top_left_p = topleft
            width_p = testWidth
            # img = cv2.GaussianBlur(img, (5,5), 0)
            # img2 = cv2.GaussianBlur(img2, (5,5), 0)
            # matchTemplate(img, img2)
            # iter = 2
            # kernel = np.ones((2,2),np.uint8)
            # closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = iter)
            # img_closing = cv2.morphologyEx(img2,cv2.MORPH_CLOSE,kernel, iterations = iter)
            # matchTemplate(closing, img_closing)
            # opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations = iter)
            # img_opening = cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel, iterations = iter)
            # matchTemplate(opening, img_opening)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    print(width_p, top_left_p)
    image_f = cv2.resize(img2.copy(), (resized(img2, width_p)))
    bottom_right_p = (top_left_p[0] + w, top_left_p[1] + h*5)
    # top_left_p = (top_left_p[0], top_left_p[1]*4)
    cv2.rectangle(image_f,top_left_p, bottom_right_p, 255, 2)
    cv2.imshow("Final", image_f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
