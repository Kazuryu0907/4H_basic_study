import cv2
import numpy as np
import math
import src.dobot

def calculateGlobal(coordinate:np.ndarray,h:float,angle:float) -> np.ndarray:
    coo:np.ndarray = np.append(coordinate,0)
    inv_A:np.ndarray = np.array([[ 0.00157501,  0.        , -0.48559031],
                      [ 0.        ,  0.00156963, -0.37898503],
                      [ 0.        ,  0.        ,  1.        ]])
    R:np.ndarray = np.array([[math.cos(-angle),-math.sin(-angle),0],[math.sin(-angle),math.cos(-angle),0],[0,0,1]])
    X:np.ndarray = h*R@inv_A@coo.T
    return X

#17*theta
#112
#m = 1.5
#端数24
#18mm/rad
#40 72
def calculateGrip(l:float) -> float:
    maxlong = 112.
    calcul = maxlong - l
    rad = calcul/18.
    deg = math.degrees(rad)/2.
    return deg


cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
_,img = cap.read()
#img:np.ndarray = cv2.imread(f"C:/Users/kazum/Desktop/p/1633703253.5631435.jpg")
# mask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),np.array([10,150,150]),np.array([50,255,255]))
# masked = cv2.bitwise_and(img,img,mask=mask)
# masked = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,img_binary = cv2.threshold(img_gray,30,200,cv2.THRESH_BINARY)

contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 100:
            x,y,w,h = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,255),2)
            M = cv2.moments(box)
            #重心計算
            center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
            cv2.drawMarker(img,center,(255,0,0))
            cv2.line(img,(img.shape[1]//2,0),(img.shape[1]//2,img.shape[0]),(0,0,255))
            cv2.line(img,(0,img.shape[0]//2),(img.shape[1],img.shape[0]//2),(0,0,255))
            cv2.imshow("before",img)
            img_center = (img.shape[0]//2,img.shape[1]//2)
            diff = np.array(list(center)) - np.array(list((img_center[1],img_center[0])))
            vec = calculateGlobal(diff,-115,0)
            #vec = np.append(vec,1)
            print(vec)
            # _,img = cap.read()


# cv2.imshow("a",img)
cv2.waitKey(0)
