import cv2
import numpy as np
import itertools

class getMarker:
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
