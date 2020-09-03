import cv2
import os

path_model_out = './test output/'
path_HR = './../../btp/super resolution data/test/HR/'
path_LR = './../../btp/super resolution data/test/LR/'
files=os.listdir(path_LR)

average = 0
tot = 0

for file in files:
    # if file != '1066.png':
    #     continue
    print('Start : ' + file)
    tot += 1
    predicted=cv2.imread(path_model_out+file)
    original_HR=cv2.imread(path_HR+file)

    psnr = cv2.PSNR(predicted , original_HR)

    current = os.path.splitext(os.path.basename(file))
    f = open(path_model_out + '../PSNRmetrics/' + current[0] + '.txt', "w")
    content = 'PSNR : ' + str(psnr)
    f.write(content)
    average += psnr
    f.close()
    print('Done : '+file)

average /= tot

f = open('./averagePSNR.txt', "w")
content = 'PSNR : ' + str(average)
f.write(content)
f.close()