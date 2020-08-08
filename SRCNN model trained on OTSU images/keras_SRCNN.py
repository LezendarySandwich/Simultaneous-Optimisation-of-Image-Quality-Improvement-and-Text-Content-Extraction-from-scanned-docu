import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as kb
import cv2
import numpy
import os

epochs=50
batch_size=16
num_training_examples=42600
num_validate_examples=9659

path_training_X='C:\\Users\\dell\\Desktop\\super resolution data\\train\\interpulated\\'
path_training_Y='C:\\Users\\dell\\Desktop\\super resolution data\\train\\otsu_image\\'
filenames_training=os.listdir(path_training_Y)

path_validate_X='C:\\Users\\dell\\Desktop\\super resolution data\\validate\\interpulated\\'
path_validate_Y='C:\\Users\\dell\\Desktop\\super resolution data\\validate\\otsu_image\\'
filenames_validate=os.listdir(path_validate_Y)

def tensor_otsu(image):
    rank = image.shape.rank
    if rank != 2 and rank != 3:
        raise ValueError("Image should be either 2 or 3-dimensional.")

    if image.dtype!=tf.int32:
        image = tf.cast(image, tf.int32)

    r, c = image.shape
    hist = tf.math.bincount(image, dtype=tf.int32)
    
    if len(hist)<256:
        hist = tf.concat([hist, [0]*(256-len(hist))], 0)

    current_max, threshold = 0, 0
    total = r * c

    spre = [0]*256
    sw = [0]*256
    spre[0] = int(hist[0])

    for i in range(1,256):
        spre[i] = spre[i-1] + int(hist[i])
        sw[i] = sw[i-1]  + (i * int(hist[i]))

    for i in range(256):
        if total - spre[i] == 0:
            break

        meanB = 0 if int(spre[i])==0 else sw[i]/spre[i]
        meanF = (sw[255] - sw[i])/(total - spre[i])
        varBetween = (total - spre[i]) * spre[i] * ((meanB-meanF)**2)

        if varBetween > current_max:
            current_max = varBetween
            threshold = i

    final = tf.where(image>threshold,255,0)
    return final

def my_loss(y_actual,y_predicted):
    loss=kb.mean(kb.sum(kb.sum(tensor_otsu(y_predicted)-y_actual)))
    return loss

class MyDataLoader(Sequence):

    def __init__(self,path_X,path_Y,filenames,batch_size):
        self.path_X=path_X
        self.path_Y=path_Y
        self.filenames=filenames
        self.batch_size=batch_size
     
    def __len__(self):
        return numpy.ceil(len(self.filenames) / float(self.batch_size)).astype(numpy.int)

    def __getitem__(self,id):
        selected_files=self.filenames[id*batch_size:(id+1)*batch_size]
        return numpy.array([cv2.imread(self.path_X+file_name,0).reshape(128,128,1).astype(numpy.float) for file_name in selected_files]),numpy.array([cv2.imread(self.path_Y+file_name,0).reshape(128,128,1).astype(numpy.float) for file_name in selected_files])

model = Sequential()
model.add(Conv2D(64, kernel_size=9, padding='same', activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(0.0001), input_shape=(128,128,1)))
model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(0.0001)))
model.add(Conv2D(1, kernel_size=5, padding='same', activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(0.0001)))
adam=Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

training_data=MyDataLoader(path_training_X,path_training_Y,filenames_training,batch_size)
validate_data=MyDataLoader(path_validate_X,path_validate_Y,filenames_validate,batch_size)

model.fit_generator(generator=training_data,steps_per_epoch=(num_training_examples//batch_size),epochs=epochs,verbose=1,validation_data=validate_data,validation_steps=(num_validate_examples//batch_size))

model_json = model.to_json()
with open("model_SRCNN.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_weights_SRCNN.h5")
print("Saved model to disk")