import cv2
from functions import getContours,getRolledRect
import numpy as np
img = cv2.imread("tweak.png")
contours = getContours(img,np.array([0,100,100]),np.array([10,255,255]))
for cnt in contours:
    box = getRolledRect(cnt)
    cv2.drawContours(img,[box],-1,(0,0,255))
cv2.imshow("a",img)
cv2.waitKey(0)