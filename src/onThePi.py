import numpy as np
import cv2
from communication import I2c
def objectIsCatched(img:np.ndarray,minLineLength:int=100) -> bool:
    imgOriginal = img
    hsvLower = np.array([330/2., 30*2.55, 30*2.55])  # 抽出色の下限
    hsvUpper = np.array([360/2., 100*2.55, 100*2.55])  # 抽出色の上限
    raserResult = hsvExtraction(imgOriginal, hsvLower, hsvUpper)
    raserGry = cv2.cvtColor(raserResult, cv2.COLOR_BGR2GRAY)
    ret, raserBin = cv2.threshold(raserGry, 20, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(raserBin, rho=1, theta=np.pi /
                            180, threshold=40, minLineLength=minLineLength, maxLineGap=3)
    if lines is not None:#直線検知された場合
    #     for line in lines:#linesの数だけ回る　回れ
    #         imgRedLine=imgOriginal.copy()
    #         print(line)
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(imgRedLine, (x1, y1), (x2, y2), (0, 255, 0), 3)#直線を緑の線で描画　変数名はきにすんな
    #     cv2.imshow('line', imgRedLine)
        return False
    
    else:#print("把持検知")
        return True

# HSVで特定の色を抽出する関数
def hsvExtraction(image, hsvLower, hsvUpper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    result = cv2.bitwise_and(image, image, mask=hsv_mask)  # 元画像とマスクを合成
    return result

def wait4catched(cap,comm:I2c,datas:list) -> None:
    _,img = cap.read()
    while 1:
        datas[1] += 10
        print(datas[1])
        comm.send(datas)
        flag = 1 << 2
        fir = True
        predata = None
        results = 0
        count = 0
        while 1:
            _,img = cap.read()
            results += 1 if objectIsCatched(img) else 0
            count += 1.
            data = comm.request()
            if predata != None and data != predata:
                fir = False
            if not fir and (flag & data) >> 2:
                break
            predata = data
        result = results/count
        print(result)
        if result > 0.6:
            return 1
# img = cv2.imread(f"C:\\Users\\kazum\\Documents\\GitHub\\cutted\\tri\\IMG_20210719_170754.jpg")
# objectIsCatched(cv2.resize(img,(640,480)))
# cv2.waitKey(0)