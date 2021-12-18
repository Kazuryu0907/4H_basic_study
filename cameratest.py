import cv2
img = cv2.imread("..//sqmask//IMG_20210719_164418.jpg",0)
img2 = cv2.imread("..//sqmask//IMG_20210706_172808.jpg",0)
cv2.imshow("a",img)
cv2.imshow("b",img2)
cv2.waitKey(0)