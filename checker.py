import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

mtx = np.load("mtx.npy")
dist = np.load("dist.npy")
capture = cv2.VideoCapture(0)
while True:

    # 画像の取得
    ret, img = capture.read()
    height = img.shape[0]
    width = img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("before",gray)

    newcameramtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),1,(width,height))

    dst = cv2.undistort(gray,mtx,dist,None,newcameramtx)

    x,y,w,h = roi
    dst = dst[y:y+h,x:x+w]
    cv2.imshow("after",dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()