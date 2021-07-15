import cv2

aruco = cv2.aruco

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
img = cv2.imread("IMG_20210705_162224.jpg")

img = cv2.resize(img,(img.shape[1]//5,img.shape[0]//5))

corners,ids,rejectedImgPoints = aruco.detectMarkers(img,dictionary)

aruco.drawDetectedMarkers(img,corners,ids,(0,255,0))

cv2.imshow("result.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()