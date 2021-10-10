import cv2
import time
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    stime = time.time()
    ret,frame = cap.read()
    cv2.imshow("frame",frame)
    print("Display FPS = ", 1.0 / (time.time() - stime))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()