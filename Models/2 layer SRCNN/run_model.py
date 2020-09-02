import cv2
import numpy
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D
from keras.utils import Sequence
from keras.optimizers import Adam
import os

test_data_path='C:\\Users\\dell\\Desktop\\super resolution data\\test\\interpulated\\'

json_file = open('model_SRCNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model

loaded_model.load_weights("model_weights_SRCNN.h5")
print("Loaded model from disk")


for file in os.listdir(test_data_path):

    input=numpy.array([cv2.imread(test_data_path+file,0).reshape(128,128,1).astype(numpy.float)])

    output=loaded_model.predict(input)
    output=numpy.rint(output[0].reshape(128,128)).astype(numpy.int)

    for i in range(0,128):
        for j in range(0,128):
            if(output[i][j]<0):
                output[i][j]=0
            if(output[i][j]>255):
                output[i][j]=255

    cv2.imwrite('.\\test output\\'+file,output) 
    print('Done :: '+file)