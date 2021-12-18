import numpy as np
import math
import cv2
from functions import getContours,getRolledRect
from PIL import Image
def getrotationMatrix(th):
    rad = np.deg2rad(th)
    M = np.array([[math.cos(rad),-math.sin(rad)],
                  [math.sin(rad),math.cos(rad)]])
    return M
def getleftestcoor(arr):
    x = 1e+10
    x_i = 0
    for i,ar in enumerate(arr):
        if ar[0] < x:
            x = ar[0]
            x_i = i
    return arr[x_i]

def cv22square2pill(im):
    pill_im = im.copy()
    pill_im = pill_im[:,:,::-1]
    pill_im = Image.fromarray(pill_im) 
    pill_im = expand2square(pill_im,(0,0,0))
    pill_im = pill_im.resize((250,250))
    cv2_im = np.array(pill_im,dtype=np.uint8)
    cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_RGB2BGR)
    return cv2_im
def getlong(box):
    return([int(math.sqrt(1.*(box[0][0]-box[1][0])**2+1.*(box[0][1]-box[1][1])**2)),int(math.sqrt(1.*(box[1][0]-box[2][0])**2+1.*(box[1][1]-box[2][1])**2))])

def extendimg(arr):
    y,x = list(arr.shape)
    arr = np.insert(arr,x//2,arr[x//2,:],axis=1)
    #cv2.imshow("d",arr)

def getconvexside(arr):
    y,x = list(arr.shape)
    per = 0.3
    
    per_y  = int(y*per)
    per_x = int(x*per)

    up = arr[0:per_y,0:x]
    down = arr[y-per_y:y,0:x]
    right = arr[0:y,x-per_x:x]
    left = arr[0:y,0:per_x]
    way = [up.sum(),down.sum(),left.sum(),right.sum()]
    maxval = 0
    maxi = 0
    for i,val in enumerate(way):
        if val > maxval:
            maxval = val
            maxi = i
    return(maxi,per_x,per_y)

def getcatchcenter(arr):
    i,per_x,per_y = getconvexside(arr)
    x,y = list(arr.shape)
    center = [0,0]#y,x
    if i == 3:#up
        center[0] = per_y//2
        center[1] = x//2
    elif i == 2:
        center[0] = (2*y-per_y)//2
        center[1] = x//2
    elif i == 1:#left
        center[0] = y//2
        center[1] = per_x//2
    elif i == 0:
        center[0] = y//2
        center[1] = (2*x-per_x)//2
    return(tuple(center))

def getrotatecoor(arr,th):
    center = np.array(list(getcatchcenter(arr)))
    M = getrotationMatrix(th)
    coor = np.dot(M,center)
    coor = np.int0(coor)
    return(coor)



def p_tile_threshold(img_gry, per):
    """
    Pタイル法による2値化処理
    :param img_gry: 2値化対象のグレースケール画像
    :param per: 2値化対象が画像で占める割合
    :return img_thr: 2値化した画像
    """

    # ヒストグラム取得
    img_hist = cv2.calcHist([img_gry], [0], None, [256], [0, 256])

    # 2値化対象が画像で占める割合から画素数を計算
    all_pic = img_gry.shape[0] * img_gry.shape[1]
    pic_per = all_pic * per

    # Pタイル法による2値化のしきい値計算
    p_tile_thr = 0
    pic_sum = 0

    # 現在の輝度と輝度の合計(高い値順に足す)の計算
    for hist in img_hist:
        pic_sum += hist

        # 輝度の合計が定めた割合を超えた場合処理終了
        if pic_sum > pic_per:
            break

        p_tile_thr += 1

    # Pタイル法によって取得したしきい値で2値化処理
    ret, img_thr = cv2.threshold(img_gry, p_tile_thr, 255, cv2.THRESH_BINARY)

    return img_thr
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

img = cv2.imread("C:\\Users\\kazum\\Desktop\\63.jpg")
contours = getContours(img,np.array([10,100,100]),np.array([50,255,255]))
maxcnt = max(contours,key=lambda cnt:cv2.contourArea(cnt))
x,y,w,h = cv2.boundingRect(maxcnt)
rect = cv2.minAreaRect(maxcnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
# center = getleftestcoor(box)
M = cv2.moments(box)
center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
RM2d = cv2.getRotationMatrix2D((int(center[0]),int(center[1])),rect[2],1)
img_rot = cv2.warpAffine(img,RM2d,((int(img.shape[1]),int(img.shape[0]))))
mask = cv2.inRange(cv2.cvtColor(img_rot,cv2.COLOR_BGR2HSV),np.array([10,100,100]),np.array([50,255,255]))
masked = cv2.bitwise_and(img_rot,img_rot,mask=mask)
#masked = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)

cv2.imshow("a",masked)
cv2.waitKey(0)