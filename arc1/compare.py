import numpy 
import cv2
import os
import pytesseract

path_LR="C:\\Users\\dell\\Desktop\\super resolution data\\test\\LR\\"
path_interpulated="C:\\Users\\dell\\Desktop\\super resolution data\\test\\interpulated\\"
path_model_out=".\\test output\\"
path_HR="C:\\Users\\dell\\Desktop\\super resolution data\\test\\HR\\"
write_path=".\\comparisions\\"

label_image=cv2.imread("C:\\Users\\dell\\Desktop\\super resolution data\\label_banner.png")

files=os.listdir(path_LR)

def tesseract_box(image,color):
    boxes=pytesseract.image_to_boxes(image)
    for b in boxes.splitlines():
        b=b.split(' ')
        image=cv2.rectangle(image,(int(b[1]),128-int(b[2])),(int(b[3]),128-int(b[4])),color,2)
    return image

for file in files:
    original_LR=cv2.imread(path_LR+file)
    original_LR=cv2.copyMakeBorder(original_LR,32,32,32,32,cv2.BORDER_CONSTANT,value=(155,155,155))

    interpulated=cv2.imread(path_interpulated+file)

    predicted=cv2.imread(path_model_out+file)

    original_HR=cv2.imread(path_HR+file)

    ocr_predicted=cv2.imread(path_model_out+file)
    ocr_predicted=tesseract_box(ocr_predicted,(11,237,255))

    ocr_original=cv2.imread(path_HR+file)
    ocr_original=tesseract_box(ocr_original,(85,244,77))

    image_stack=numpy.hstack((original_LR,interpulated,predicted,original_HR,ocr_predicted,ocr_original))
    final_image=numpy.vstack((image_stack,label_image))

    cv2.imwrite(write_path+file,final_image)
    print("Done :: "+file)