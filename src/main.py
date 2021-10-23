import dobot
import cv2
import numpy as np
import itertools
import math
from communication import I2c,Serial_local
from getMarker import Marker

def getRoll(box):
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
    print(f"longtan:{atanangle}")
    #角度調整-90 ~ 90
    if atanangle < -90:
        atanangle += 180
    elif atanangle > 90:
        atanangle -= 180
    return [math.sqrt(l1) if l1 > l2 else math.sqrt(l2),atanangle]

def dobotSetup():
    db.set_cordinate_speed(velocity=60,jerk=6)
    db.set_jump_pram(height=60,zlimit=185)
    db.jump_joint_to(j1=0,j2=0,j3=100,j4=0)

#Raspiカメラの画像からglobal座標へ変換
def calculateGlobal(coordinate,h,angle):
    inv_A = np.array([[ 0.00157501,  0.        , -0.48559031],
                      [ 0.        ,  0.00156963, -0.37898503],
                      [ 0.        ,  0.        ,  1.        ]])
    R = np.array([[math.cos(-angle),-math.sin(-angle),0],[math.sin(-angle),math.cos(-angle),0],[0,0,1]])
    X = h*R@inv_A@coordinate.T
    return X

"""
チャート
　動く->一定の高さで止める->ismovingがFalse->カメラとる->中心へ移動->
"""
def chart():
    h = 100
    db.move(db.JUMP_TO,(0,0,h,0))
    while db.ismoving:
        continue
    cap.read()
    #画像から重心取得
    calculateGlobal()
    db.move(db.JUMP_TO,(0,0,h,0))



ser_local = Serial_local("COM4",9600)
#17*theta
#112
#m = 1.5
#端数24
#18mm/rad
#40 72
def calculategrip(l):
    maxlong = 112.
    calcul = maxlong - l
    rad = calcul/18.
    deg = math.degrees(rad)/2.
    return deg

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)


if __name__ == "__main__":
    #dobotのインスタンス作成
    db = dobot.CommandSender("192.168.33.40",8889)
    #dobotの初期設定
    dobotSetup()
    while db.ismoving:
        continue
    #time.sleep(1)
    #マーカーのインスタンス作成
    datas = [0,0,0,0]
    ADDR = 0x1E
    #i2c = I2c(ADDR)
    mk = Marker()
    #マーカー間のサイズ定義
    mapSize = ((400-13)*1,(465-13)*1)
    mapSize = (286,438)
    #cap_img = cv2.imread("../sample.jpg")
    _,cap_img = cap.read()
    cv2.imshow("A",cap_img)
    cv2.waitKey(10)
    #マーカー読み取り及び射影変換
    img = mk.getCorner(cap_img,mapSize,input_ids=[0,1,2,3],middle=[4,5])

    #恥物の色のマスク
    mask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),np.array([10,150,150]),np.array([50,255,255]))
    masked = cv2.bitwise_and(img,img,mask=mask)
    masked = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)

    #輪郭取得
    img_gray = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    ret,img_binary = cv2.threshold(img_gray,30,200,cv2.THRESH_BINARY)
    cv2.imshow("mask",img_binary)
    contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        #小さいのは除外
        if cv2.contourArea(cnt) > 100:
            #回転あり最小矩形
            #send_local("1,0")
            x,y,w,h = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(img,[box],0,(0,0,255),2)
            M = cv2.moments(cnt)
            #重心計算
            center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))

            cv2.drawMarker(img,center,(255,0,0))
            #cv2座標からdobot座標系への変換
            t_x = center[0] - img.shape[1]//2
            t_center = (t_x,center[1])
            print(t_center)
            longtan,roll = getRoll(box)
            gripdeg = calculategrip(longtan)
            #print(roll,gripdeg)
            datas[1] = roll
            #動かす(自作関数(rで角度取得))
            db.move(db.JUMP_TO,[t_center[1],t_center[0],100,-int(roll)])
            ser_local.send("1"+","+str(int(gripdeg)))
            ser_local.wait4Servo(1)
            #i2c.send(datas)
            #print(t_center[1],t_center[0],100,-int(getRoll(box)))           
    cv2.imshow("a",img)
    cv2.imwrite("before.jpg",img)
    cv2.waitKey(0)