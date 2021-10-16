import cv2
import numpy as np
import glob
import pprint
square_size = 1.8
pattern_size = (7,7)

pattern_points = np.zeros((np.prod(pattern_size),3),np.float32)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1,2)
pattern_points *= square_size

objpoints = []
imgpoints = []
files = glob.glob(f"C:\\Users\\kazum\\Desktop\\p\\*.jpg")
for f in files:
    img = cv2.imread(f)
    print(img.shape)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,corner = cv2.findChessboardCorners(gray,pattern_size)

    if ret:
        print("d")
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,30,0.1)
        cv2.cornerSubPix(gray,corner,(5,5),(-1,-1),term)
        imgpoints.append(corner.reshape(-1,2))
        objpoints.append(pattern_points)

    #cv2.imshow("img",img)

print("calcu")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.save("mtx",mtx)
np.save("dist",dist.ravel())
pprint.pprint(ret)
pprint.pprint(mtx)
pprint.pprint(dist)