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

if __name__ == "__main__":
    mk = Marker()
    mapSize = ((450-13)*1,(300-13)*1)
    img = mk.getCorner(cv2.imread(f"../tt.jpg"),mapSize,input_ids=[2,3,1,0])
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,img_binary = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,contours,-1,(0,0,255))
    import math
    import dobot

    db = dobot.CommandSender("192.168.33.40",8893)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x,y,w,h = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print(box)
            p1 = box[0]
            p2 = box[1]
            #print(math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))
            p1 = box[1]
            p2 = box[2]
            #print(math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))
            cv2.drawContours(img,[box],0,(0,0,255),2)
            M = cv2.moments(cnt)
            center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
            if rect[2] != 90.0:
                cv2.drawMarker(img,center,(255,0,0))
                t_x = center[0] - mapSize[1]//2
                t_center = (t_x,center[1])
                print(t_center)
                if t_x > 0:
                    #0:left
                    db.arm_orientation(0)
                else:
                    db.arm_orientation(1)
                db.jump_to(t_center[0],t_center[1],60,int(rect[2]))
                #db.go_to(t_center[0],t_center[1],60,int(rect[2]))
            #print(rect[2])
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.imshow("a",img)
    cv2.waitKey(0)