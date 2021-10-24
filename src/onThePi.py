import numpy as np
import cv2

def objectIsCatched(img:np.ndarray,minLineLength:int=100) -> bool:
    imgOriginal = img
    hsvLower = np.array([330/2., 30*2.55, 30*2.55])  # 抽出色の下限
    hsvUpper = np.array([360/2., 100*2.55, 100*2.55])  # 抽出色の上限
    raserResult = hsvExtraction(imgOriginal, hsvLower, hsvUpper)
    lower_white = np.array([0,0,150])
    upper_white = np.array([180,60,255])
    r = hsvExtraction(imgOriginal,lower_white,upper_white)
    cv2.imshow("a",cv2.cvtColor(cv2.cvtColor(r+raserResult,cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY))
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

img = cv2.imread(f"C:\\Users\\kazum\\Documents\\GitHub\\cutted\\tri\\IMG_20210719_170754.jpg")
objectIsCatched(cv2.resize(img,(640,480)))
cv2.waitKey(0)