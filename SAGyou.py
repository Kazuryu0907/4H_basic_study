import cv2
import numpy as np

img = cv2.imread("b.png")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_,bin_img = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(bin_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

index = [i for i,hi in enumerate(hierarchy[0]) if hi[2] == -1]

contours = [cnt for i,cnt in enumerate(contours) if i in index]
sort = [cv2.contourArea(cnt) for cnt in contours]
sort = np.array(sort)
sort = np.argsort(sort)[-1::-1]
contours = [cnt for i,cnt in enumerate(contours) if i in sort[:2:]]
print(contours)
cv2.drawContours(img,contours,-1,color=(0,0,255),thickness=3)
cv2.imshow("A",img)

cv2.waitKey(0)