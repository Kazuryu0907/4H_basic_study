from communication import Serial_local,I2c
from enum import IntEnum

class CommIndex(IntEnum):
  TIP = 0
  HAND = 1
  GRID = 2
  LINE = 3

comm = Serial_local("COM4",9600)

data = f"{int(CommIndex.TIP)},90"
comm.send(data)
while 1:
  comm.update()
  comm.wait4Servo(int(CommIndex.TIP))
  print("ServoEND")
  break
"""
comm = I2c(0x1E)
data = [0,90,0,0]
comm.send(data)
while 1:
  comm.wait4Servo(1)
  print("ServoEND")
  break
"""