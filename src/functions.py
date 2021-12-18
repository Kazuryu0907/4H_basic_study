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
    cv2.imshow("mask",img_binary)
    contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def getRolledRect(cnt:np.ndarray) -> np.ndarray:
    # x,y,w,h = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def img_diff(img1,img2,img3,th):
    diff1 = cv2.absdiff(img1,img2)
    diff2 = cv2.absdiff(img2,img3)
    diff = cv2.bitwise_and(diff1,diff2)
    diff[diff < th] = 0
    diff[diff >= th] = 255
    mask = cv2.medianBlur(diff,5)
    cv2.imshow("a",mask)
    return mask

def wait4move(cap):
    frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    cnt = 0
    while cap.isOpened():
        mask = img_diff(frame1,frame2,frame3,10)
        moment = cv2.countNonZero(mask)
        if moment < 100:
            cnt += 1
        else:
            cnt = 0
        if cnt > 5:
            print("End moving")
            return 0
        frame1 = frame2
        frame2 = frame3
        frame3 = cv2.cvtColor(cap.read()[1],cv2.COLOR_RGB2GRAY)

def tweakCoor(cap,color,h=-127) -> np.ndarray:
    _,img = cap.read()
    hsv_max = np.array([50,255,255]) if color == "Y" else np.array([50,255,255])
    hsv_min = np.array([10,150,100]) if color == "Y" else np.array([10,90,50])
    contours = getContours(img,hsv_min,hsv_max)
    maxcnt = max(contours,key=lambda cnt:cv2.contourArea(cnt))
    cnt = contours[maxcnt]
    box = getRolledRect(cnt)
    M = cv2.moments(box)
    center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
    img_center = (img.shape[0]//2,img.shape[1]//2)
    diff = np.array(list(center)) - np.array(list((img_center[1],img_center[0])))
    vec = calculateGlobal(diff,h,-math.pi/2.)
    roll = getRoll(box)[1]
    if roll > 45:
        roll = 90 - roll
    diffcoor = np.array([vec[1],vec[0],0,int(roll)+90])
    diffcoor *= -1
    return diffcoor

if __name__ == "__main__":
    box = np.array([[0,0],[0,10],[10,1],[0,1]])
    l,roll = getRoll(box)
    print(l,roll+90)