import multiprocessing as mp
import multiprocessing.sharedctypes
import numpy as np
import cv2
import time

WIDTH = 640
HEIGHT = 480
def camera_reader(out_buf,buf1_rdy):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    while(1):
        try:
            #cap_stime = time.time()
            ret,img = cap.read()
            if ret is False:
                raise IOError
            buf1_rdy.clear()
            memoryview(out_buf).cast("B")[:] = memoryview(img).cast("B")[:]
            buf1_rdy.set()
            #print("Capture+Conversion+Copy FPS = ", 1.0 / (time.time() - cap_stime))

        except KeyboardInterrupt:
            break
    cap.release()
if __name__ == "__main__":
    buf1 = multiprocessing.sharedctypes.RawArray("B",WIDTH*HEIGHT*3)
    buf1_rdy = mp.Event()
    buf1_rdy.clear()
    p1 = mp.Process(target=camera_reader,args=(buf1,buf1_rdy),daemon=True)
    p1.start()

    caped_img = np.empty((HEIGHT,WIDTH,3),dtype=np.uint8)
    while True:
        try:
            dis_stime = time.time()
            buf1_rdy.wait()
            caped_img[:,:,:] = np.reshape(buf1,(HEIGHT,WIDTH,3))
            buf1_rdy.clear()
            cv2.imshow("f",caped_img)
            key = cv2.waitKey(1)
            print(key)
            #print("Display FPS = ", 1.0 / (time.time() - dis_stime))
        except KeyboardInterrupt:
            print("Waiting camera reader to finish.")
            p1.join(10)
            break
    
    cv2.destroyAllWindows()