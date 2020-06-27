
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import sys
from pytesseract import Output
sys.path.append("C:\\Program Files\\Tesseract-OCR\\tessdata\\")
sys.path.append("C:\\Program Files\\Tesseract-OCR\\")
tessdata_dir_config = r'--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata\\" -l eng --oem 1 --psm 6'
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'

def blacken_img(img,img_col):


    ret,img = cv2.threshold(img, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = cv2.GaussianBlur(img,(5,5),0)

    #img=cv2.bitwise_not(img)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    #cv2.imshow('img', cv2.resize(img,(500,500)))
    #cv2.waitKey(0)
    n_boxes = len(d['level'])
    cot=0
    flak=0
    #print(d['text'])
    tik=0
    cols=[i.strip().lower() for i in d['text']]
    if 'birth' in cols or 'birth:' in cols:
        tik=1
    flagg=[]
    counter=1
    for i in list(range(n_boxes))[::-1]:
        if len(d['text'][i])==4 and d['text'][i].isdigit():
            flagg.append(counter)
            counter+=1
        else:
            flagg.append(-1)
            counter=1
    flagg=flagg[::-1]
    cuss=0
    for i,fg in zip(range(n_boxes),flagg):
        #print(i)
        if fg==2:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(img_col, (x, y), (x + w, y + h), (0, 0, 0), -1)
            (x, y, w, h) = (d['left'][i-1], d['top'][i-1], d['width'][i-1], d['height'][i-1])
            cv2.rectangle(img_col, (x, y), (x + w, y + h), (0, 0, 0), -1)
            cuss=1
    #print(flagg)
    if cuss==0 and 0:
        print(d['text'])
        print(flagg)
    return img_col,cot,cuss





import os

imgs_dirr = 'C:\\Users\\subatheeshkaleidofin\\Aadhar Masking\\TestFront\\'
failure_dir = 'C:\\Users\\subatheeshkaleidofin\\Aadhar Masking\\Result3\\'
success_dir = 'C:\\Users\\subatheeshkaleidofin\\Aadhar Masking\\Result3\\'

# imgs_dirr="C:\\Users\\trivikramkaleidofin\\Desktop\\datasets\\github\\NachNAadhaar\\aadhar card images\\"
# failure_dir="C:\\Users\\trivikramkaleidofin\\Desktop\\datasets\\github\\NachNAadhaar\\modified_new_samples_fail_2\\"
# success_dir="C:\\Users\\trivikramkaleidofin\\Desktop\\datasets\\github\\NachNAadhaar\\modified_new_samples_pass_2\\"

to_count=0
skip_count=0
for k in os.listdir(imgs_dirr):
    try:
        print("Skipped:{},Total:{},Success_Percentage: {}%".format(skip_count,to_count,(1.0-(skip_count/to_count))*100.0))
    except:
        print("Skipped:0,Total:0,Success_Percentage: 100%")
    to_count+=1
    #k='3776b13e-9a1b-4b7f-917f-1a35b49ed14b.jpg'
    #k="33b64eca-a2bc-4ff6-b813-f2c17d6eb74c.jpg"
    #k='12de3b96-3e3c-422a-a489-54377291c52b.jpg'
    #k='0d3d68c5-b291-47c2-be04-323d762778cf.jpg'
    re=cv2.imread(imgs_dirr+k,0)
    re_col=cv2.imread(imgs_dirr+k)
    #re_bor = cv2.copyMakeBorder(re,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
    re2,cnt,success=blacken_img(re.copy(),re_col.copy())
    #print(cnt)
    #print('Saving..')
    if success==0:
        skip_count+=1
        cv2.imwrite("{}{}".format(failure_dir,k),re2)
        continue
    else:
        #cv2.imshow('img', cv2.resize(re2,(500,500)))
        #cv2.waitKey(0)
        cv2.imwrite("{}{}".format(success_dir,k),re2)
    #break
