import cv2
import numpy as np
from communication import I2c,Serial_local
from functions import *
from getMarker import Marker
from onThePi import objectIsCatched
import dobot

#----------list of instancce
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
commu = Serial_local("COM4",9600)
db = dobot.CommandSender("192.168.33.40",8889)
mk = Marker()
#--------------------------------

#----------list of global variable
datas = [0,0,0,0]
ADDR = 0x1E
#マーカー間のサイズ定義
#mapSize = ((400-13)*1,(465-13)*1)
mapSize = (286,438)
#----------------------
if __name__ == "__main__":
    #dobotの初期設定
    db.dobotSetup()
    #cap_img = cv2.imread("../sample.jpg")
    _,cap_img = cap.read()
    cv2.imshow("A",cap_img)
    cv2.waitKey(10)
    #マーカー読み取り及び射影変換
    img = mk.getCorner(cap_img,mapSize,input_ids=[0,1,2,3],middle=[4,5])
    #マスク・輪郭取得
    contours = getContours(img,np.array([10,150,150]),np.array([50,255,255]))
    for cnt in contours:
        #小さいのは除外
        if cv2.contourArea(cnt) > 100:
            #回転あり最小矩形
            box = getRolledRect(cnt)

            cv2.drawContours(img,[box],0,(0,0,255),2)
            M = cv2.moments(box)
            #重心計算
            center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))

            cv2.drawMarker(img,center,(255,0,0))
            #cv2座標からdobot座標系への変換
            t_x = center[0] - img.shape[1]//2
            t_center = (t_x,center[1])
            print(t_center)
            long,roll = getRoll(box)
            print(f"long:{long}")
            gripdeg = calculateGrip(long)
            datas[1] = roll
            #動かす(自作関数(rで角度取得))
            db.move(db.JUMP_TO,[t_center[1],t_center[0],100,-int(roll)])
            commu.send("1"+","+str(int(gripdeg)))
            commu.wait4Servo(1)
            #i2c.send(datas)
            #print(t_center[1],t_center[0],100,-int(getRoll(box)))           
    cv2.imshow("a",img)
    #cv2.imwrite("before.jpg",img)
    cv2.waitKey(0)
