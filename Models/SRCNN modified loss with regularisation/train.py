import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
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