import cv2
import multiprocessing
import multiprocessing.sharedctypes
import time
import numpy

WIDTH=640
HEIGHT=480

def camera_reader(out_buf, buf1_ready):
  capture = cv2.VideoCapture(0)
  """try:
    capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
  except TypeError:
    capture = cv2.VideoCapture(0)

  if capture.isOpened() is False:
    raise IOError

  if isinstance(capture.get(cv2.CAP_PROP_CONVERT_RGB), float):
    capture.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
  else:
    capture.set(cv2.CAP_PROP_CONVERT_RGB, False)
    """
  capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
  capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
  capture.set(cv2.CAP_PROP_FPS, 30)

  while(True):
    try:
      capture_start_time = time.time()
      ret, frame = capture.read()
      if ret is False:
        raise IOError
      #print("Capture FPS = ", 1.0 / (time.time() - capture_start_time))
      bgr_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
      #cv2.imshow('frame2', bgr_frame)
      buf1_ready.clear()
      memoryview(out_buf).cast('B')[:] = memoryview(bgr_frame).cast('B')[:]
      buf1_ready.set()
      print("Capture+Conversion+Copy FPS = ", 1.0 / (time.time() - capture_start_time))
    except KeyboardInterrupt:
      # 終わるときは CTRL + C を押す
      break
  capture.release()

if __name__ == "__main__":
  buf1 = multiprocessing.sharedctypes.RawArray('B', HEIGHT*WIDTH*3)
  buf1_ready = multiprocessing.Event()
  buf1_ready.clear()
  p1=multiprocessing.Process(target=camera_reader, args=(buf1,buf1_ready), daemon=True)
  p1.start()

  captured_bgr_image = numpy.empty((HEIGHT, WIDTH, 3), dtype=numpy.uint8)
  while True:
    try:
      display_start_time = time.time()
      buf1_ready.wait()
      captured_bgr_image[:,:,:] = numpy.reshape(buf1, (HEIGHT, WIDTH, 3))
      buf1_ready.clear()
      cv2.imshow('frame', captured_bgr_image)
      cv2.waitKey(1)
      print("Display FPS = ", 1.0 / (time.time() - display_start_time))
    except KeyboardInterrupt:
      # 終わるときは CTRL + C を押す
      print("Waiting camera reader to finish.")
      p1.join(10)
      break

  cv2.destroyAllWindows()