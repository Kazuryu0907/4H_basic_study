class catchError:
    def __init__(self) -> None:
        self.msg = ""

    def error(self,i:int) -> None:
        if i == 1:
            self.msg = "camera load Error"
        elif i == 2:
            self.msg = "I2C Error"
        elif i == 3:
            self.msg = "dobot connect Error"
        elif i == 4:
            self.msg = "can't find Marker"