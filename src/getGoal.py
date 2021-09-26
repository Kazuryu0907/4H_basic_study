import cv2
import numpy as np

#==============未DEBUG============

class getGoal:
    """
    画像から目標地点の輪郭(内側)の座標を取得する

    Attributes
    ----------
    getGoal : np.ndarray
        座標を取得する
    """
    def __init__(self):
        self.a = ""
    def getGoal(self,img:np.ndarray,color:str,goalnums:int,thre:list,hsv_min_max:list=None) -> np.ndarray:
        """
        目標地点の輪郭の座標を取得する

        Parameters
        ----------
        img : np.darray
            元画像をcv2.imreadしたもの
        color : str
            マスクを掛ける色
            "white"か"green"
        goalnums : int
            目標地点の個数
        thre : list
            二値化時の閾値 ex.[thre0,thre1]

        以下必須ではない
        
        hsv_min_max : list
            マスク時の詳細設定 ex.[(min),(max)]
        
        Returns
        -------
        contours : np.ndarray
            目標地点の輪郭(内側)座標
        """
        self._img = img
        self._color = color
        self.colors()
        if not hsv_min_max == None:
            self._hsv_min ,self._hsv_max = hsv_min_max
        self._goalnums = goalnums
        self.mask()
        _,self._bin_img = cv2.threshold(cv2.cvtColor(self._masked,cv2.COLOR_BGR2GRAY),thre[0],thre[1],cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(self._bin_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)


        index = [i for i,hi in enumerate(hierarchy[0]) if hi[2] == -1]
        contours = [cnt for i,cnt in enumerate(contours) if i in index]
        sort = [cv2.contourArea(cnt) for cnt in contours]
        sort = np.array(sort)
        sort = np.argsort(sort)[-1::-1]
        contours = [cnt for i,cnt in enumerate(contours) if i in sort[:goalnums:]]
        return contours

    def mask(self):
        mask = cv2.inRange(cv2.cvtColor(self._img,cv2.COLOR_BGR2HSV),self._hsv_min,self._hsv_max)
        masked = cv2.bitwise_and(self._img,self._img,mask=mask)
        self._masked = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)

    def colors(self):
        if self.colors == "white":
            self._hsv_min = np.array([0,0,200])
            self._hsv_max = np.array([180,45,255])
        elif self.colors == "green":
            self._hsv_min = np.array([30,64,0])
            self._hsv_max = np.array([90,255,255])
