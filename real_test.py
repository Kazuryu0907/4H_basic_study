import cv2
import numpy as np

import math
def getRoll(box):
    ymin = [0,0]
    xmax = [0,0]
    xmin = [0,0]
    for i in range(4):
        if i != 0:
            ymin = box[i] if box[i][1] < ymin[1] else ymin[:]
            xmax = box[i] if box[i][0] > xmax[0] else xmax[:]
            xmin = box[i] if box[i][0] < xmin[0] else xmin[:]
        else:
            ymin = box[i][:]
            xmax = box[i][:]
            xmin = box[i][:]

    p0 = ymin
    p1 = xmax
    p2 = xmin
    print(p0,p1,p2)
    cv2.drawMarker(img,p0,(255,0,0))
    cv2.drawMarker(img,p1,(0,255,0))
    cv2.drawMarker(img,p2,(0,0,255))
    #平行の法
    tan1 = math.atan2(p0[1]-p1[1],p0[0]-p1[0])
    l1 = (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2
    tan2 = math.atan2(p0[1]-p2[1],p0[0]-p2[0])
    l2 = (p0[0]-p2[0])**2 + (p0[1]-p2[1])**2
    longtan = tan1 if l1 > l2 else tan2
    if l1 > l2:
        cv2.line(img,p0,p1,(255,0,0),thickness=3)
    else:
        cv2.line(img,p0,p2,(255,0,0),thickness=3)
    atanangle = math.degrees(longtan)
    
    #-90 ~ 90

    if atanangle < -90:
        atanangle += 180
    elif atanangle > 90:
        atanangle -= 180
    print(f"longtan:{atanangle}")
    if atanangle > 45 or atanangle < -45:
        print("縦")
    else:
        print("横")
    cv2.putText(img,str(round(atanangle,3)),p0,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
    
    """
    if l1 > l2:
        #横長
        print("横",angle)
        roll = angle-90
    else:
        #縦長
        print("縦",angle)
        if angle < 45:
            roll = 90-angle
        else:
            roll = angle
    """
    return atanangle

img = cv2.imread("img.jpg")

mask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),np.array([10,150,150]),np.array([50,255,255]))
masked = cv2.bitwise_and(img,img,mask=mask)
masked = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)

img_gray = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
ret,img_binary = cv2.threshold(img_gray,30,200,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        x,y,w,h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(getRoll(box))

cv2.imshow("A",img)
cv2.waitKey(0)