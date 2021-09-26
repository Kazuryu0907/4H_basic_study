from PIL.Image import TRANSPOSE
import cv2
import numpy as np

def getrad(cnt):
    for c in cnt:
        x,y = c.T
        print(x)
def mask(img,hsv_min,hsv_max,thre=None):
    mask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),hsv_min,hsv_max)
    masked = cv2.bitwise_and(img,img,mask=mask)
    masked = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)
    if thre == None:
        _,img_binary = cv2.threshold(cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _,img_binary = cv2.threshold(cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY),thre[0],thre[1],cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(img_binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    #rect = cv2.minAreaRect(contours)
    cnts = []
    for cnt in contours:
        if 100 < cv2.contourArea(cnt):
            cnts.append(cnt)
    for cnt in cnts:
        """
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        """
        getrad(cnt)
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        print(len(approx))
        cv2.drawContours(img,[approx],-1,(0,0,255),thickness=2)
    cv2.imshow("a",img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()


