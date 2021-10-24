import serial
import pigpio

class I2c:
    def __init__(self,ADDR:bytes):
        self._addr = ADDR
        self._pi = pigpio.pi()
        self._id = self._pi.i2c_open(1,ADDR)
    def send(self,datas:list) -> None:
        self._pi.i2c_write_i2c_block_data(self._id,0,datas)
    def request(self) -> bytes:
        #http://abyz.me.uk/rpi/pigpio/python.html#i2c_read_device
        return self._pi.i2c_read_byte_data(self._id,self._addr)
    def wait4Servo(self,index:int) -> None:
        flag:bytes = 0 | 1 << index
        while 1:
            data = self.request()
            if flag == data:
                break

class Serial_local:
    def __init__(self,port:str,rate:int) -> None:
        self.ser = serial.Serial(port,rate)
    def update(self):
        if self.ser.inWaiting() > 0:
            self._data = self.ser.read()
            self.ser.flushInput()
    def send(self,str:str) -> None:
        self.ser.write(str.encode("utf-8"))
    def getdata(self) -> bytes:
        return self._data
    def wait4Servo(self,index:int):
        flag:bytes = 0 | 1 << index
        while 1:
            self.update()
            data = self.getdata()
            if data == flag:
                break