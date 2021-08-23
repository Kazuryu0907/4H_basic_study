import cv2
import numpy as np

aruco = cv2.aruco

dic = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
img = cv2.imread("../data/IMG_20210705_162224.jpg")
img = cv2.resize(img,(img.shape[1]//5,img.shape[0]//5))

corners,ids,rejectedImgPoints = aruco.detectMarkers(img,dic)

W = 1000
H = 800
output_corners = [[0,0],[W,0],[0,H],[W,H]]

markers = [0,1,3,2]
corner_index = [0,1,3,2]
#corner_index = [2,3,1,0]
input_corners = []
ids = np.ravel(np.array(ids))
print(ids)
for i in range(4):
    try:
        index = ids.tolist().index(markers[i])
    except:
        print("What is this marker?")
    input_corners.append(corners[index][0][corner_index.index(i)].tolist())

print(type(input_corners),type(output_corners))
M = cv2.getPerspectiveTransform(np.array(input_corners,dtype=np.float32),np.array(output_corners,dtype=np.float32))
rst = cv2.warpPerspective(img,M,(W,H))

cv2.imshow("result.png",rst)
cv2.waitKey(0)
cv2.destroyAllWindows()
