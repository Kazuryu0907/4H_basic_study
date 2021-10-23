import cv2
import numpy as np
import itertools
import math
from communication import I2c,Serial_local
class Marker:
    """
    画像からマーカー内部の画像をトリミングする

    Attributes
    ----------
    getCorner : np.ndarray
        トリミングする
    """
    def __init__(self):
        self.aruco = cv2.aruco
        self.dictionary = self.aruco.getPredefinedDictionary(self.aruco.DICT_4X4_50)
    
    def getCorner(self,img:np.ndarray,size:tuple,input_ids=None,tri=False,middle=None) -> np.ndarray:
        """
        マーカーからトリミングする

        Parameters
        ----------
        img : np.darray
            元画像をcv2.imreadしたもの
        size : tuple
            トリミング後の縦x横

        以下必須ではない
        
        input_ids : list
            マーカーのidを左上から時計周りに配列化したもの(int)
            例 [0,1,2,3]
        tri : bool
            tri=Trueの時マーカーの内側をトリミングする

        middle : list
            中心測定用のマーカーid
            2個を想定
        
        Returns
        -------
        img : np.ndarray
            トリミング後の画像
        """
        
        self.img = img
        self.img = cv2.resize(self.img,(self.img.shape[1]//1,self.img.shape[0]//1))
        #マーカー読み取り
        self.corners,self.ids,self.rejectedImgPoints = self.aruco.detectMarkers(self.img,self.dictionary)
        #一次元にしてる
        self.ids = list(itertools.chain.from_iterable(self.ids))
        if input_ids is None:
            #id指定されてなかったら
            self.getPosition()
            if self.getPositionFlag:
                return None
        else:
            #id指定されてたら
            self.input_ids = input_ids

        #座標取得
        self.getInputcorners(tri)
        self.H,self.W = size
        self.output_corner = np.array([[0,0],[self.W,0],[self.W,self.H],[0,self.H]],dtype=np.float32)
        
        #射影変換
        self.M = cv2.getPerspectiveTransform(self.input_corner,self.output_corner)
        self.rst = cv2.warpPerspective(self.img,self.M,(self.W,self.H))
        #中心調整
        if middle is not None:
                middle_coors = [self.corners[self.ids.index(i)] for i in middle]
                middle_coors_points = [coo[0][i] for i,coo in enumerate(middle_coors)]
                #マーカーのindexの0と1を取得
                to_cvt = np.zeros((len(middle_coors_points),3),dtype=np.float32)
                #C = 0
                for i,coo in enumerate(middle_coors_points):
                    #C += coo[0]
                    to_cvt[i,:2] = coo[:]
                    to_cvt[i,2] = 1
                #C = int(C//2)
                #cv2.line(self.img,(C,0),(C,self.H),(0,0,255))
                #cv2.imwrite("CENTER.jpg",self.img)
                #射影変換
                cvt_middle_coors = [self.M@coo.T for coo in to_cvt]
                centerX = 0
                for coor in cvt_middle_coors:
                    centerX += coor[0]
                #射影変換後の中心座標計算
                centerX = centerX / 2.
                print(f"centerX:{centerX},middle:{self.W/2}")
                offset = int(centerX - self.W/2.)
                #画像サイズ拡張＆平行移動
                tx = offset if offset > 0 else 0
                M = np.array([[1,0,tx],[0,1,0]],dtype=np.float32)
                dst = cv2.warpAffine(self.rst,M,(self.W+int(abs(offset)),self.H))
                cv2.line(dst,(dst.shape[1]//2,0),(dst.shape[1]//2,self.H),(0,0,255))
                return dst
        cv2.line(self.rst,(self.rst.shape[1]//2,0),(self.rst.shape[1]//2,self.H),(0,0,255))
        return self.rst
    
    #0-------1
    #| index |
    #3-------2
    #コーナーのどの点をとるかをいい感じに取得してる
    def getInputcorners(self,tri=False) -> list:
        self.input_corner = np.empty((0,2),float)
        for i,id in enumerate(self.input_ids):
            index = self.ids.index(id)
            if tri:
                #内側トリミング
                index = (index + 2)%4
            self.input_corner = np.append(self.input_corner,np.array([self.corners[index][0][i]],dtype=float),axis=0)
        self.input_corner = self.input_corner.astype(np.float32)
    #各座標からコーナー４つを振り分け
    #自分で作ったのにもはや何してるかわからん
    def getPosition(self) -> None:
        self.getPositionFlag = False
        table = {}
        self.input_ids = [0,0,0,0]
        aveX = 0
        aveY = 0
        for i in range(4):
            try:
                id = self.ids[i]
                corner = self.corners[i][0][0]
                table[str(id)] = corner
                aveX += corner[0]
                aveY += corner[1]
            except:
                self.getPositionFlag = True
                return 0
        aveX /= 4.
        aveY /= 4.
        for id,(x,y) in table.items():
            id = int(id)
            if x < aveX:
                if y < aveY:
                    self.input_ids[0] = id
                else:
                    self.input_ids[3] = id
            else:
                if y < aveY:
                    self.input_ids[1] = id
                else:
                    self.input_ids[2] = id

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

#Raspiカメラの画像からglobal座標へ変換
def calculateGlobal(coordinate:np.ndarray,h:float,angle:float) -> np.ndarray:
    coo:np.ndarray = np.append(coordinate,0)
    inv_A:np.ndarray = np.array([[ 0.00157501,  0.        , -0.48559031],
                      [ 0.        ,  0.00156963, -0.37898503],
                      [ 0.        ,  0.        ,  1.        ]])
    R:np.ndarray = np.array([[math.cos(-angle),-math.sin(-angle),0],[math.sin(-angle),math.cos(-angle),0],[0,0,1]])
    X:np.ndarray = h*R@inv_A@coo.T
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



#17*theta
#112
#m = 1.5
#端数24
#18mm/rad
#40 72
def calculateGrip(l):
    maxlong = 112.
    calcul = maxlong - l
    rad = calcul/18.
    deg = math.degrees(rad)/2.
    return deg

import dobot
import math

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ser_local = Serial_local("COM4",9600)

if __name__ == "__main__":
    #dobotのインスタンス作成
    db = dobot.CommandSender("192.168.33.40",8889)
    #dobotの初期設定
    db.dobotSetup()
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
            M = cv2.moments(box)
            #重心計算
            center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))

            cv2.drawMarker(img,center,(255,0,0))
            #cv2座標からdobot座標系への変換
            t_x = center[0] - img.shape[1]//2
            t_center = (t_x,center[1])
            print(t_center)
            longtan,roll = getRoll(box)
            gripdeg = calculateGrip(longtan)
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
