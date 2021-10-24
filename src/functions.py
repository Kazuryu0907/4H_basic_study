import cv2
import numpy as np
import math

def getRoll(box:np.ndarray) -> list:
    ymin = box[box[:,1].argmin()]
    xmax = box[box[:,0].argmax()]
    xmin = box[box[:,0].argmin()]

    p0 = ymin
    p1 = xmax
    p2 = xmin
    #辺の長さ取得
    tan1 = math.atan2(p0[1]-p1[1],p0[0]-p1[0])
    l1 = (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2
    tan2 = math.atan2(p0[1]-p2[1],p0[0]-p2[0])
    l2 = (p0[0]-p2[0])**2 + (p0[1]-p2[1])**2
    #長辺の角度を代入
    longtan = tan1 if l1 > l2 else tan2
    #rad to degree
    atanangle = math.degrees(longtan)
    #角度調整-90 ~ 90
    if atanangle < -90:
        atanangle += 180
    elif atanangle > 90:
        atanangle -= 180
    return [math.sqrt(l1) if l1 > l2 else math.sqrt(l2),atanangle]

#Raspiカメラの画像からglobal座標へ変換
def calculateGlobal(coordinate:np.ndarray,h:float,angle:float) -> np.ndarray:
    coo:np.ndarray = np.append(coordinate,0)
    inv_A:np.ndarray = np.array([[ 0.00157501,  0.        , -0.48559031],
                      [ 0.        ,  0.00156963, -0.37898503],
                      [ 0.        ,  0.        ,  1.        ]])
    R:np.ndarray = np.array([[math.cos(-angle),-math.sin(-angle),0],[math.sin(-angle),math.cos(-angle),0],[0,0,1]])
    X:np.ndarray = h*R@inv_A@coo.T
    return X

#m = 1.5
#端数24
def calculateGrip(l:float) -> float:
    maxlong = 112.
    calcul = maxlong - l
    rad = calcul/18.
    deg = math.degrees(rad)/2.
    return deg

def getContours(img:np.ndarray,hsv_min:np.ndarray,hsv_max:np.ndarray) -> np.ndarray:
    mask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),hsv_min,hsv_max)
    masked = cv2.bitwise_and(img,img,mask=mask)
    masked = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)
    img_gray = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    ret,img_binary = cv2.threshold(img_gray,30,200,cv2.THRESH_BINARY)
    # cv2.imshow("mask",img_binary)
    contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def getRolledRect(cnt:np.ndarray) -> np.ndarray:
    # x,y,w,h = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box
