from PIL import Image
import numpy as np
import glob
from keras.preprocessing import image
import os
import random,math
import cv2

output_dir = ""
def draw_images(generater,x,dir_name,index):
    save_name = "extened-"+str(index)
    g = generater.flow(x,batch_size=1,save_to_dir=output_dir,save_prefix=save_name,save_format="jpeg")

    for i in range(50):
        batch = g.next()
"""
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def read_img(filename):
    image = Image.open(filename)
    image = expand2square(image,(0,0,0))
    image = image.resize((250,250))
    #image = image.convert("L")
    #image = np.array(image,dtype=np.uint8)

    return image

files = glob.glob("*.jpg")

for f in files:
    print(f)
    img = read_img(f)
    img.save("data/"+f)
"""


datagen = image.ImageDataGenerator(rotation_range=90,
                                    width_shift_range=20,
                                    height_shift_range=20,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    vertical_flip=True)

files = ["^","L"]
output_dir = "ado"
for f in files:
    output_dir = "data/"+"ado-"+f
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    images = glob.glob("data/"+f+"/"+"*.jpg")
    for i in range(len(images)):
        img = image.load_img(images[i])
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        #draw_images(datagen,x,output_dir,i)

X = []
Y = []
categories = ["^","L"]

def make_data(files):
    global X,Y
    X = []
    Y = []
    for cat,fname in files:
        add_data(cat,fname)
    return np.array(X),np.array(Y)

def add_data(cat,fname):
    img = cv2.imread(fname)
    #img = cv2.cvtColor(img,cv2.COLORRGB)
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

allfiles = []

for idx,cat in enumerate(categories):
    image_dir = "data/"+cat
    files = glob.glob(image_dir+"/*.jpg")
    for f in files:
        allfiles.append((idx,f))
    image_dir = "data/"+"ado-"+cat
    files = glob.glob(image_dir+"/*.jpeg")
    for f in files:
        allfiles.append((idx,f))

print(len(allfiles))

random.shuffle(allfiles)
th = math.floor(len(allfiles)*0.8)
train = allfiles[:th]
test = allfiles[th:]
X_train,y_train = make_data(train)
X_test,y_test = make_data(test)
xy = (X_train,X_test,y_train,y_test)
if not(os.path.exists("saves/")):
    os.mkdir("saves/")
np.savez("saves/tea_data.npy",X_train,X_test,y_train,y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)