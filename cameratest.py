import cv2
import numpy as np
import math
import pprint


def caluclateGlobal(coordinate,inv_A,h,angle):
    R = np.array([[math.cos(-angle),-math.sin(-angle),0],[math.sin(-angle),math.cos(-angle),0],[0,0,1]])
    print(coordinate.shape)
    X = h*R@inv_A@coordinate.T
    return X
img = cv2.imread(f"C:\\Users\\kazum\\Desktop\\p\\LaserWithYObject.jpg")
A = np.load("mtx.npy")
inv_A = np.linalg.inv(A)
pprint.pprint(inv_A)
pprint.pprint(caluclateGlobal(np.array([30,0,1]),inv_A,60,math.radians(90)))

b = np.array([[634.91574962,   0.        , 308.30893476],
       [  0.        , 637.0939446 , 241.44906484],
       [  0.        ,   0.        ,   1.        ]])

