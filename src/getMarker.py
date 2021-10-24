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
