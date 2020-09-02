import numpy as np
import cv2
import pytesseract
import os
import Metric 

path_model_out = './test output/'
path_HR = './../../btp/super resolution data/test/HR/'
path_LR = './../../btp/super resolution data/test/LR/'
files=os.listdir(path_LR)

average = [0,0]
tot = 0

for file in files:
    # if file != '1066.png':
    #     continue
    print('Start : ' + file)
    tot += 1
    predicted=cv2.imread(path_model_out+file)
    original_HR=cv2.imread(path_HR+file)

    boxesPredicted=pytesseract.image_to_boxes(predicted)
    boxesHR=pytesseract.image_to_boxes(original_HR)
    # c1 r1 c2 r2
    BOX = []
    BOX1 = []
    for b in boxesPredicted.splitlines():
        b=b.split(' ')
        BOX.append([b[0],float(b[1]),float(b[2]),float(b[3]),float(b[4])])
    for b in boxesHR.splitlines():
        b=b.split(' ')
        BOX1.append([b[0],float(b[1]),float(b[2]),float(b[3]),float(b[4])])
    # print(BOX)
    # print(BOX1)
    cur_Metric = Metric.Metrics(BOX , BOX1)
    # print(cur_Metric)
    current = os.path.splitext(os.path.basename(file))
    f = open(path_model_out + '../Metrics/' + current[0] + '.txt', "w")
    content = 'Bipartite Metric : ' + str(cur_Metric[0]) + '\nFlowMetric : ' + str(cur_Metric[1])
    f.write(content)
    average[0] += cur_Metric[0]
    average[1] += cur_Metric[1]
    f.close()
    print('Done : '+file)

average[0] /= tot
average[1] /= tot

f = open('./averageMetric.txt', "w")
content = 'Bipartite Metric : ' + str(average[0]) + '\nFlowMetric : ' + str(average[1])
f.write(content)
f.close()