import serial
import smbus

class I2c:
    def __init__(self,ADDR:bytes):
        self._addr = ADDR
        self._bus = smbus.SMBus(1)
        self.data = [0,0,0,0]
        # self._id = self._pi.i2c_open(1,ADDR)
    def send(self,datas:list) -> None:
        # self._pi.i2c_write_i2c_block_data(self._id,0,datas)
        self._bus.write_block_data(self._addr,0,datas)
    def request(self) -> bytes:
        #http://abyz.me.uk/rpi/pigpio/python.html#i2c_read_device
        # return self._pi.i2c_read_byte_data(self._id,self._addr)
        return self._bus.read_byte(self._addr)
    def wait4Servo(self,index:int) -> None:
        flag = 1 << index
        fir = True
        predata = None
        while 1:
            data = self.request()
            if predata != None and data != predata:
                fir = False
            if not fir and (flag & data) >> index:
                break
            predata = data
    def upload(self,index,data):
        self.data[index] = data
        self.send(self.data)

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
        flag:bytes = 1 << index
        fir = True
        predata = None
        while 1:
            self.update()
            data = self.getdata()
            if predata != None and data != predata:
                fir = False
            if not fir and (flag & data) >> index:
                break
            predata = data
