from sys import dont_write_bytecode
import cv2
import numpy as np
import itertools

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
    
    def getCorner(self,img:np.ndarray,size:tuple,input_ids=None,tri=False) -> np.ndarray:
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
        
        Returns
        -------
        img : np.ndarray
            トリミング後の画像
        """
        
        self.img = img
        self.img = cv2.resize(self.img,(self.img.shape[1],self.img.shape[0]))
        self.corners,self.ids,self.rejectedImgPoints = self.aruco.detectMarkers(self.img,self.dictionary)
        self.ids = list(itertools.chain.from_iterable(self.ids))
        if input_ids is None:
            self.getPosition()
            if self.getPositionFlag:
                return None
        else:
            self.input_ids = input_ids
        self.getInputcorners(tri)
        self.H,self.W = size
        self.output_corner = np.array([[0,0],[self.W,0],[self.W,self.H],[0,self.H]],dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.input_corner,self.output_corner)
        self.rst = cv2.warpPerspective(self.img,self.M,(self.W,self.H))
        return self.rst
    
    #0-------1
    #| index |
    #3-------2
    def getInputcorners(self,tri=False) -> list:
        self.input_corner = np.empty((0,2),float)
        for i,id in enumerate(self.input_ids):
            index = self.ids.index(id)
            if tri:
                index = (index + 2)%4
            self.input_corner = np.append(self.input_corner,np.array([self.corners[index][0][i]],dtype=float),axis=0)
        self.input_corner = self.input_corner.astype(np.float32)

    def getPosition(self):
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
    ymin = [0,0]
    xmax = [0,0]
    xmin = [0,0]
    for i in range(4):
        if i != 0:
            ymin = box[i] if box[i][1] < ymin[1] else ymin[:]
            xmax = box[i] if box[i][0] > xmax[0] else xmax[:]
            xmin = box[i] if box[i][0] < xmin[0] else xmin[:]
        else:
            ymin = box[i][:]
            xmax = box[i][:]
            xmin = box[i][:]

    p0 = ymin
    p1 = xmax
    p2 = xmin
    print(p0,p1,p2)
    #平行の法
    tan1 = math.atan2(p0[1]-p1[1],p0[0]-p1[0])
    l1 = (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2
    tan2 = math.atan2(p0[1]-p2[1],p0[0]-p2[0])
    l2 = (p0[0]-p2[0])**2 + (p0[1]-p2[1])**2
    longtan = tan1 if l1 > l2 else tan2
    atanangle = math.degrees(longtan)
    print(f"longtan:{atanangle}")
    #-90 ~ 90
    if atanangle < -90:
        atanangle += 180
    elif atanangle > 90:
        atanangle -= 180
    
    if atanangle > 45 or atanangle < -45:
        print("縦")
    else:
        print("横")
    
    """
    if l1 > l2:
        #横長
        print("横",angle)
        roll = angle-90
    else:
        #縦長
        print("縦",angle)
        if angle < 45:
            roll = 90-angle
        else:
            roll = angle
    """
    return atanangle

def dobotSetup():
    db.set_cordinate_speed(velocity=60,jerk=6)
    db.set_jump_pram(height=60,zlimit=185)
    db.jump_joint_to(j1=0,j2=0,j3=60,j4=0)


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if __name__ == "__main__":
    import dobot
    db = dobot.CommandSender("192.168.33.40",8889)
    dobotSetup()
    import time
    import math
    time.sleep(1)
    mk = Marker()
    mapSize = ((400-13)*1,(465-13)*1)
    #cap_img = cv2.imread("camera.jpg")
    _,cap_img = cap.read()
    img = mk.getCorner(cap_img,mapSize,input_ids=[0,1,2,3])
    #cv2.imwrite("camera.jpg",cap_img)
    mask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),np.array([10,150,150]),np.array([50,255,255]))
    masked = cv2.bitwise_and(img,img,mask=mask)
    masked = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)
    
    img_gray = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    ret,img_binary = cv2.threshold(img_gray,30,200,cv2.THRESH_BINARY)
    cv2.imshow("mask",img_binary)
    contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img,contours,-1,(0,0,255))
    import math
    #cv2.imwrite("img.jpg",img)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x,y,w,h = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print(box)
            if cv2.contourArea(cnt) > 10000:
                length = 0
                for i in range(4):
                    p1 = box[i]
                    p2 = box[(i+1)%4]
                    length = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
                cv2.putText(img,str(length),(p1[0]-30,p1[1]-30),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1.5,color=(255,255,255),thickness=1,lineType=cv2.LINE_4)
            cv2.drawContours(img,[box],0,(0,0,255),2)
            M = cv2.moments(cnt)
            center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
            if rect[2] != 810:
                cv2.drawMarker(img,center,(255,0,0))
                t_x = center[0] - mapSize[1]//2
                t_center = (t_x,center[1])
                print(t_center)
                db.arm_orientation(1 if t_x > 0 else 0)
                db.jump_to(x=t_center[1],y=t_center[0],z=100,r=-int(getRoll(box)))
    cv2.imshow("a",img)
    cv2.waitKey(0)


