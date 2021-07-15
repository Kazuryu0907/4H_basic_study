from keras import models
from keras.preprocessing import image
import numpy as np

model = models.model_from_json(open("saves/tea_predict.json").read())

model.load_weights("saves/tea_predict.hdf5")
def predict_pic(data):

    x = np.expand_dims(data,axis=0)

    features = model.predict(x)
    #print(features)

    if features[0,0] >= 0.9:
        return 0    #^
    elif features[0,1] >= 0.9:
        return 1    #L
    else:
        return -1   #other
