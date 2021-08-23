
from keras import layers,models
from keras import optimizers
import numpy as np
from keras.utils import np_utils
from numpy.core.fromnumeric import shape

from tensorflow.python.client import device_lib
from tensorflow.python.keras.callbacks import ModelCheckpoint

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(250,250,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(2,activation="sigmoid"))
model.save_fig("ex.svg")
#model.summary()
"""
model.compile(loss="binary_crossentropy",optimizer=optimizers.RMSprop(lr=1e-4),metrics=["acc"])

categories = ["^","L"]
nb_classes = len(categories)
df = np.load("saves/tea_data.npy.npz",allow_pickle=True)

X_train = df["arr_0"]
X_test = df["arr_1"]
y_train = df["arr_2"]
y_test = df["arr_3"]

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255



print(X_train.shape,y_train.shape)
y_train = np_utils.to_categorical(y_train,nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)
hdf5_file = "saves/tea_predict.hdf5"
cp = ModelCheckpoint(hdf5_file,monitor="val_loss",verbose=1,save_best_only=True,save_weights_only=True)

model = model.fit(X_train,
                y_train,
                epochs=50,
                batch_size=6,
                callbacks=[cp],
                validation_data=(X_test,y_test))

json_string = model.model.to_json()
open("saves/tea_predict.json","w").write(json_string)
"""