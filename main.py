from src import *
import cv2
mk = Marker()
tri = mk.getCorner(cv2.imread(r"C:\Users\kazum\Desktop\cutted\src\sample.jpg"),(800,1200),[0,1,2,3])

import numpy as np
#green
hsv_min = np.array([30,64,0])
hsv_max = np.array([90,255,255])
"""
hsv_min = np.array([0,0,100])
hsv_max = np.array([180,45,255])
"""
mask = cv2.inRange(cv2.cvtColor(tri,cv2.COLOR_BGR2HSV),hsv_min,hsv_max)
masked = cv2.bitwise_and(tri,tri,mask=mask)
masked = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)
ret,img_binary = cv2.threshold(cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY),0,140,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    print(cv2.contourArea(cnt))
rect = cv2.minAreaRect(contours[2])
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(tri,[box],-1,(0,0,255),thickness=2)


cv2.imshow("a",tri) 
cv2.waitKey(0)
cv2.destroyAllWindows()