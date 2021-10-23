#

import numpy as np
import cv2
from time import sleep


def objectIsCatched(filename):

    filename = 'LaserWithYObject.jpg'

    imgOriginal = cv2.imread(filename)
    imgBlur = cv2.GaussianBlur(imgOriginal, (3, 3), 0)

    hsvLower = np.array([5/2, 30*2.55, 30*2.55])  # 抽出色の下限
    hsvUpper = np.array([50/2, 100*2.55, 100*2.55])  # 抽出色の上限
    targetResult = hsvExtraction(imgBlur, hsvLower, hsvUpper)  # 出力画像

    hsvLower = np.array([330/2, 30*2.55, 30*2.55])  # 抽出色の下限
    hsvUpper = np.array([360/2, 100*2.55, 100*2.55])  # 抽出色の上限
    raserResult = hsvExtraction(imgOriginal, hsvLower, hsvUpper)

    raserGry = cv2.cvtColor(raserResult, cv2.COLOR_BGR2GRAY)
    ret, raserBin = cv2.threshold(raserGry, 20, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(raserBin, rho=1, theta=np.pi /
                            180, threshold=40, minLineLength=100, maxLineGap=3)
                            
    #print(lines)
    if lines is not None:#直線検知された場合
        #for line in lines:#linesの数だけ回る　回れ
            #imgRedLine=imgOriginal.copy()
            #x1, y1, x2, y2 = line[0]
            #imgRedLine = cv2.line(imgRedLine, (x1, y1), (x2, y2), (0, 255, 0), 3)#直線を緑の線で描画　変数名はきにすんな
        #cv2.imshow('line', imgRedLine)
        #print("非把持")
        return False
    
    else:#print("把持検知")
        return True
    #imgGry = cv2.cvtColor(hsvResult, cv2.COLOR_BGR2GRAY)

    #ret, bin_img = cv2.threshold(imgGry, 20, 255, cv2.THRESH_BINARY)

    # contours, hierarchy = cv2.findContours(
    #    bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )
    #cv2.drawContours(imgOriginal, contours, -1, color=(0, 0, 255), thickness=2)

    #result = imgOriginal
    #cv2.imshow('original', imgOriginal)
    #cv2.imshow('target', targetResult)
    #cv2.imshow('raser', raserResult)
    #cv2.imshow('rasergry', raserBin)
    

    #while True:
    #    # キー入力を1ms待って、keyが「q」だったらbreak　こいつがいないと　#表示 は一瞬で消える
    #    key = cv2.waitKey(1) & 0xff
    #    if key == ord('q'):
    #        break

    #cv2.destroyAllWindows()  # ウィンドウを閉じる


# HSVで特定の色を抽出する関数
def hsvExtraction(image, hsvLower, hsvUpper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    result = cv2.bitwise_and(image, image, mask=hsv_mask)  # 元画像とマスクを合成
    return result

