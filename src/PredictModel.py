from keras import models
from keras.preprocessing import image
import numpy as np

class PredictModel:
    def __init__(self,path:str,hdf5:str):
        self.model = models.model_from_json(open(path).read())
        self.model.load_weights(hdf5)
    
    def predict(self,img:np.ndarray) -> str:
        x = np.expand_dims(img,axis=0)

        self.features = self.model.predict(x)

        if self.features[0,0] >= 0.9:
            return "^"
        elif self.features[0,1] >= 0.9:
            return "L"
        else:
            return None
