import cv2
import numpy as np
aruco = cv2.aruco

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
img = cv2.imread(r"C:\Users\kazum\Desktop\cutted\src\sample.jpg")

img = cv2.resize(img,(img.shape[1]//5,img.shape[0]//5))

corners,ids,rejectedImgPoints = aruco.detectMarkers(img,dictionary)
print(ids,corners[1][0])

input_corners = np.array([[229.,216.], [874.,201.], [155.,666.], [950.,661.]],dtype=np.float32)
W = 1000
H = 800
output_corners = np.array([[0,0],[W,0],[0,H],[W,H]],dtype=np.float32)
M = cv2.getPerspectiveTransform(input_corners,output_corners)
rst = cv2.warpPerspective(img,M,(W,H))
#cv2.imwrite("rest.png",rst)

aruco.drawDetectedMarkers(img,corners,ids,(0,255,0))
print(type(rst))
cv2.imshow("result.png",img)
cv2.imshow("r",rst)
cv2.waitKey(0)
cv2.destroyAllWindows()
