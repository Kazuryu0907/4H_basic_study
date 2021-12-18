import cv2
import numpy as np
from communication import I2c
from functions import *
from getMarker import Marker
from getGoal import getGoal
from onThePi import wait4catched
import dobot
from enum import IntEnum
import time
#----------list of instancce
cap = cv2.VideoCapture(0)
cap_w = cv2.VideoCapture(1)
commu = I2c(0x1E)
commu.send([0,0,0,0])
db = dobot.CommandSender("192.168.33.40",8889)
mk = Marker()
gg = getGoal()
#--------------------------------
def fgoal(goal):
    approx_contours = []
    for i, cnt in enumerate(goal):
        # 輪郭の周囲の長さを計算する。
        arclen = cv2.arcLength(cnt, True)
        # 輪郭を近似する。
        approx_cnt = cv2.approxPolyDP(cnt, epsilon=0.01 * arclen, closed=True)
        approx_contours.append(approx_cnt)
        # 元の輪郭及び近似した輪郭の点の数を表示する。
        print(f"contour {i}: before: {len(cnt)}, after: {len(approx_cnt)}")
    mingoal = min(approx_contours,key=lambda ac:len(ac))
    center = [0.,0.]
    for c in mingoal:
        x,y = c[0]
        center[0] += x
        center[1] += y
    center[0] /= len(mingoal)
    center[1] /= len(mingoal)
    center[0] = center[0] - img.shape[1]//2
    return list(map(int,center))
#----------list of global variable
datas = [0,0,0,0]
ADDR = 0x1E
#マーカー間のサイズ定義
#mapSize = ((400-13)*1,(465-13)*1)
mapSize = (450,300*3)
H = 52+40
class mode(IntEnum):
    TIP = 0
    HAND = 1
    GRID = 2
    LINE = 3
targetCounter = 0
#----------------------
if __name__ == "__main__":
    #cameraの初期設定
    cap_w.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc("M","J","P","G"))
    cap_w.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_w.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #dobotの初期設定
    db.dobotSetup(H)
    _,cap_img = cap_w.read()
    #マーカー読み取り及び射影変換
    img = mk.getCorner(cap_img,mapSize,input_ids=[0,1,2,3])
    #マスク・輪郭取得
    contours = getContours(img,np.array([10,150,150]),np.array([50,255,255]))
    # Bcontours = getContours(img,np.array([10,90,50]),np.array([50,255,255]))
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
            # print(t_center)
            long,roll = getRoll(box)
            print(f"long:{long}")
            print(t_center[1],t_center[0],H,-int(roll))
            #動かす(自作関数(rで角度取得))
            tarcoor = [t_center[1],t_center[0],H,-int(roll)]
            db.move(db.JUMP_TO,tarcoor)
            print(tarcoor)
            commu.upload(mode.HAND,int(calculateGrip(long)))
            # commu.wait4Servo(2)
            wait4move(cap)
            try:
                diffcoor = tweakCoor(cap,"Y")
            except:
                diffcoor = [0,0,0,0]
            tarcoor:np.ndarray = np.array(db.currentPosition) + diffcoor
            tarcoor = tarcoor.astype(np.int64)
            #caribration
            db.move(db.JUMP_TO,tarcoor.tolist())
            wait4move(cap)
            commu.upload(mode.GRID,1)
            time.sleep(1)
            commu.upload(mode.GRID,0)
            commu.upload(mode.LINE,1)
            commu.data[:] = wait4catched(cap,commu,datas)
            commu.upload(mode.LINE,0)
            # mask = cv2.imread("mask.png",cv2.IMREAD_GRAYSCALE)
            # img[mask==0] = [0,0,0]
            # goal = gg.getGoal(img,"white",2,[30,200])
            # center = fgoal(goal)
            center = [-150,250+(-15*targetCounter)]
            # center = [-150,250+(-40*targetCounter)]
            db.move(db.JUMP_TO,[center[1],center[0],H,0])
            wait4move(cap)
            commu.upload(mode.HAND,0)
            #commu.wait4Servo(2)

    #cv2.imshow("a",img)
    #cv2.imwrite("before.jpg",img)
    #cv2.waitKey(0)
 
 
"""
 import cv2
from onThePi import objectIsCatched
from communication import I2c

comm = I2c(0x1E)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
_,img = cap.read()
datas = [0,0,0,1]
comm.send(datas)
#cv2.imshow("a",img)
while objectIsCatched(img):
    datas[1] += 10
    #cv2.imshow("a",img)
    #cv2.waitKey(10)
    print(datas[1])
    comm.send(datas)
    comm.wait4Servo(2)
    _,img = cap.read()
#cv2.waitKey(0)

"""
