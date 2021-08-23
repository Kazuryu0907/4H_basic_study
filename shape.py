import cv2
import numpy as np
from numpy.core import einsumfunc
from PIL import Image
import predict
import math

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

#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
#cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc("H","2","6","4"))
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
while True:
    #_, img = cap.read()


    img = cv2.imread(r"C:\Users\kazum\Desktop\Camera\IMG_20210421_145906.jpg",1)
    #ret,img = cap.read()
    img = cv2.resize(img,(int(img.shape[1]/5),int(img.shape[0]/5)))
    img_original = img.copy()
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,img_binary = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    cv2.imshow("A",img_binary)
    contours,hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #laplacian = cv2.Laplacian(img_gray,cv2.CV_64F)
    #print("var:{}".format(laplacian.var()))
    

    color = (255,255,0)
    imgs = []

    categorious = ["^","L"]

    for i,contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:
            continue

        x,y,w,h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(img,[box],0,(0,0,255),2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        center = getleftestcoor(box)
        

        RM2d = cv2.getRotationMatrix2D(tuple(center),rect[2],1)
        img_rot = cv2.warpAffine(img,RM2d,((int(img.shape[1]),int(img.shape[0]))))

        l = getlong(box)
        arr = np.array([y,y+h,x,x+w])
        im = img_original[arr[0]:arr[1],arr[2]:arr[3]]
        cv2_im = cv22square2pill(im)
        cv2.imshow("B",cv2_im)
        data = np.asarray(cv2_im)
        data = data.astype("float32") / 255
        pre = predict.predict_pic(data)
        if pre == 1:
            #print(l)
            
            trim = img_rot[center[1]:center[1]+l[0],center[0]:center[0]+l[1]]
            trim_gray = cv2.cvtColor(trim,cv2.COLOR_BGR2GRAY)
            #extendimg(np.array(trim_gray))
            trim_n = p_tile_threshold(trim_gray,0.8)
            #cv2.imshow(str(i),trim_n)
            trim_n = np.array(trim_n)
            way = ["up","down","left","right"]
            coor = getrotatecoor(trim_n,rect[2])
            globalcoor = coor+center
            cv2.circle(img,tuple(globalcoor),5,(255,0,0))
        else:
            M = cv2.moments(contour)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.circle(img,(cx,cy),5,(255,0,0))


        cv2.putText(img,categorious[pre],(arr[2],arr[0]),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3,cv2.LINE_AA)
        
    cv2.imshow("C",img)
        #cv2.imwrite("rect.jpg",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()