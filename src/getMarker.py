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
    
    def getCorner(self,img:np.ndarray,input_ids:list,size:tuple,tri=False) -> np.ndarray:
        """
        マーカーからトリミングする

        Parameters
        ----------
        img : np.darray
            元画像をcv2.imreadしたもの
        input_ids : list
            マーカーのidを左上から時計周りに配列化したもの(int)
            例 [0,1,2,3]
        size : tuple
            トリミング後の縦x横
        tri : bool
            tri=Trueの時マーカーの内側をトリミングする
        
        Returns
        -------
        img : np.ndarray
            トリミング後の画像
        """
        
        self.img = img
        self.img = cv2.resize(self.img,(self.img.shape[1]//5,self.img.shape[0]//5))
        self.corners,self.ids,self.rejectedImgPoints = self.aruco.detectMarkers(self.img,self.dictionary)
        self.getInputcorners(input_ids,tri)
        if self.input_corner is None:
            return None
        self.H,self.W = size
        self.output_corner = np.array([[0,0],[self.W,0],[self.W,self.H],[0,self.H]],dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.input_corner,self.output_corner)
        self.rst = cv2.warpPerspective(self.img,self.M,(self.W,self.H))
        return self.rst
    
    #0-------1
    #| index |
    #3-------2
    def getInputcorners(self,input_ids:list,tri=False) -> list:
        self.input_corner = np.empty((0,2),float)
        self.ids = list(itertools.chain.from_iterable(self.ids))
        for i,id in enumerate(input_ids):
            try:
                index = self.ids.index(id)
                if tri:
                    index = (index + 2)%4
                self.input_corner = np.append(self.input_corner,np.array([self.corners[index][0][i]],dtype=float),axis=0)
            except:
                self.input_corner = None
                return 0
        self.input_corner = self.input_corner.astype(np.float32)