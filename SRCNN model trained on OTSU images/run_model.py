import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import cv2
import numpy
import tensorflow
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam

test_data_path='C:\\Users\\dell\\Desktop\\super resolution data\\test\\interpulated\\'

json_file = open('model_otsu_otsu.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model

loaded_model.load_weights("model_weights_otsu_ostu.h5")
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
    output=cv2.imread('.\\test output\\'+file,0)
    threshold,output=cv2.threshold(output,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite('.\\otsu_images\\'+file,output)
    print('Done :: '+file)