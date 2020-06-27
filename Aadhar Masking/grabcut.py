import numpy as np
import cv2
import os

from matplotlib import pyplot as plt

# img = cv.imread('messi5.jpg')
# mask = np.zeros(img.shape[:2],np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# rect = (50,50,450,290)
#
# cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(),plt.show()

def resized(frame, w):
    ht, wt, ch = frame.shape
    return (w,(int)(ht/(wt/w)))



for image_name in os.listdir("Test"):
    # cv2.namedWindow("Image")
    # cv2.setMouseCallback("Image", mouseCallback)

    img = cv2.imread("Test/"+image_name, 1)
    # img = cv2.imread("TestFront/test2.jpg", 0)

    image = cv2.resize(img.copy(), (resized(img, 1000)))
    h, w, c = image.shape



    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (1,1,w,h)

    cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    cv2.imshow("dfs", mask2)
    cv2.waitKey(0)
    image = image*mask2[:,:,np.newaxis]
    plt.imshow(image),plt.colorbar(),plt.show()


    # cv2.imshow("Images", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
