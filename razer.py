import cv2
import numpy as np
import glob


for fname in glob.glob("tri/*.jpg"):
    name = fname.split(f"\\")[-1]

    img = cv2.imread(fname)
    img = cv2.resize(img,(img.shape[1]//1,img.shape[0]//1))
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower = np.array([160,0,200],dtype=np.uint8)
    upper = np.array([180,100,255],dtype=np.uint8)

    maskRed = cv2.inRange(hsv,lower,upper)
    img_red = cv2.bitwise_and(img,img,mask=maskRed)
    img_red = cv2.cvtColor(img_red,cv2.COLOR_HSV2BGR)
    img_gray = cv2.cvtColor(img_red,cv2.COLOR_BGR2GRAY)
    black = np.zeros(img.shape)
    ret, img_binary = cv2.threshold(img_gray, 150, 255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contour = cv2.drawContours(black, contours, -1, (255, 255, 255), 20)
    cv2.imwrite("masked/"+name,img_contour)
    print("masked/"+name)

cv2.waitKey(0)