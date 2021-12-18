import cv2
import numpy as np

def exec_gamma_correction( filepath, gamma ):
    # 処理対象の画像をロード
    imgS = cv2.imread(filepath)
 
    # γ値を使って Look up tableを作成
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
 
    # Look up tableを使って画像の輝度値を変更
    imgA = cv2.LUT(imgS, lookUpTable)
 
    # PILで表示用画像を作成
    from PIL import Image, ImageDraw, ImageFont
    imgA_RGB = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgP = Image.fromarray(imgA_RGB)
 
    # 画像の左上にγ値の表示を埋め込む
    obj_draw = ImageDraw.Draw(imgP)
    #obj_font = ImageFont.truetype("/usr/share/fonts/ipa/ipagp.ttf", 40)
    #obj_draw.text((10, 10), "γ = %.1f" % gamma, fill=(255, 255, 255), font=obj_font)
 
    # 表示実行
    imgP.show()

exec_gamma_correction("w.jpg",2)