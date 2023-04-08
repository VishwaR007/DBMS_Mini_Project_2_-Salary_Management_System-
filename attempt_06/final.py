from ultralytics import YOLO
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import os
import time
import shutil
import sys 
import json

model = YOLO("yolov8m-seg-custom.pt")


DIR_1 = "E:\\AIML\\ZIP_TAG_PROJECT\\attempt_06\\final_check_imgs"
DIR_run_seg_prediction_head = "E:\AIML\ZIP_TAG_PROJECT\\attempt_06\\runs\segment\predict"
DIR_run_seg_prediction_tail = "E:\AIML\ZIP_TAG_PROJECT\\attempt_06\\dimensions_output_folder"

Num_Good_Pro = 0
Num_Bad_Pro = 0
Num_Without_Head = 0
Num_Without_Tail = 0
Total_Num = 0

for img in os.listdir(DIR_1):

    path_1 = os.path.join(DIR_1, img)

    img_real = cv2.imread(path_1)

    DIR_runs_segment = "E:\\AIML\\ZIP_TAG_PROJECT\\attempt_06\\runs\\segment"
    for i in os.listdir(DIR_runs_segment):
        if i == 'predict':
            shutil.rmtree(os.path.join(DIR_runs_segment, i))



    prediction = model.predict(source = img_real, show = True, save = True, hide_labels = False, hide_conf = False, conf = 0.5, save_txt = False, save_crop = False, line_thickness = 2)


    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    # img_real = cv2.imread("E:\AIML\ZIP_TAG_PROJECT\attempt_06\final_check_imgs\1.png")
    img = cv2.resize(img_real, (700, 700))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=3)

    result_img = closing.copy()
    contours,hierachy = cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hitung_objek = 0

    pixelsPerMetric = None

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 1000 or area > 120000:
            continue

        orig = img.copy()
        box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)

        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)
    
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(orig, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 5)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 5)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 5)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 5)

        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                (255, 0, 255), 2)
        
        lebar_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        panjang_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = lebar_pixel
            pixelsPerMetric = panjang_pixel
        lebar = lebar_pixel
        panjang = panjang_pixel

        # print("L : ", format(lebar_pixel/25.5))   # ==============================================
        # print("P : ", format(panjang_pixel/25.5))     # ==============================================

        cv2.putText(orig, "L: {:.1f}CM".format(lebar_pixel/25.5),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
        cv2.putText(orig, "P: {:.1f}CM".format(panjang_pixel/25.5),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
        #cv2.putText(orig,str(area),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2)
        hitung_objek+=1


    cv2.putText(orig, "Image: {}".format(hitung_objek),(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)  
    # cv2.imshow('Kamera',orig)
    cv2.imwrite("E:\\AIML\\ZIP_TAG_PROJECT\\attempt_06\\dimensions_output_folder\\output.jpg", orig)
    cv2.imwrite("E:\WEB DEVLOPMENT\PROJECTS\projects\\test2\\src\\assets\\dimensions_output_folder\\output.jpg", orig)
    # cv2.waitKey(0)


    Total_Num += 1

    # Initilising the result variable to ZERO :     1 => GOOD    0 => BAD
    HEAD = 1

    # First Check using head condition
    if len(prediction[0].boxes) == 1:
        # print("//Head is there")    # ==============================================
        HEAD = 1
    else:
        # print("//Head is not there")    # ==============================================
        HEAD = 0
        Num_Without_Head += 1

    # Assigning values of dimensions of the body to VARIABLES and creating an ARRAY.
    L = format(lebar_pixel/25.5)
    R = format(panjang_pixel/25.5)
    arr1 = [L, R]

    # print(arr1)
    # print(type(arr1))

    TAIL = 1
    # Checking the tail condition
    for i in range(len(arr1)):
        val_of_arr1_in_int_01 = float(arr1[i])
        val_of_arr1_in_int_02 = float(arr1[i+1])

        if (val_of_arr1_in_int_01 > 12.0 or val_of_arr1_in_int_02 > 12.0) :
            # print("//Tail Present ")    # ==============================================
            TAIL = 1
        else :
            # print("//TAIL CUT")     # ==============================================
            TAIL = 0
            Num_Without_Tail += 1
        break


    # THE LAST RESULT CONDITION :
    if HEAD == 1 and TAIL == 1:
        GOOD_PRODUCT = 1
        Num_Good_Pro += 1
        # print("//GOOD PRODUCT ... :)")      # ==============================================
    else :
        GOOD_PRODUCT = 0
        Num_Bad_Pro += 1
        # print("//BAD PRODUCT ... :(")       # ==============================================





    

    sys.stdout = open('E:\\WEB DEVLOPMENT\\PROJECTS\\projects\\test2\\src\\declare.js', 'w')
    # sys.stdout = open('declare.js', 'w')

    pred_img_path_head = os.path.join(DIR_run_seg_prediction_head, 'image0.jpg')
    img_rep_path_head = pred_img_path_head.replace('\\', '/')
    shutil.copy(pred_img_path_head, 'E:\\WEB DEVLOPMENT\\PROJECTS\\projects\\test2\\src\\assets\\predict\\')
    # jsonobj_head_img = json.dumps(img_rep_path_head)
    jsonobj_head_img = json.dumps("../../assets/predict/image0.jpg")
    print("export const jsonobj_head_img = '{}'".format(jsonobj_head_img))

    pred_img_path_tail = os.path.join(DIR_run_seg_prediction_tail, 'output.jpg')
    img_rep_path_tail = pred_img_path_tail.replace('\\', '/')
    shutil.copy(pred_img_path_tail, 'E:\WEB DEVLOPMENT\PROJECTS\projects\\test2\src\\assets\\dimensions_output_folder\\')
    # jsonobj_tail_img = json.dumps(img_rep_path_tail)
    jsonobj_tail_img = json.dumps("../../assets/dimensions_output_folder/output.jpg")
    print("export const jsonobj_tail_img = '{}'".format(jsonobj_tail_img))

    jsonobj_HEAD_value = json.dumps(HEAD)
    print("export const jsonobj_HEAD_value = '{}'".format(jsonobj_HEAD_value))
    jsonobj_TAIL_value = json.dumps(TAIL)
    print("export const jsonobj_TAIL_value = '{}'".format(jsonobj_TAIL_value))
    # jsonobj_TAIL_value = json.dumps(TAIL)
    # print("export const jsonobj_TAIL_value = '{}'".format(jsonobj_TAIL_value))
    jsonobj_GOODpRODUCT_value = json.dumps(GOOD_PRODUCT)
    print("export const jsonobj_GOODpRODUCT_value = '{}'".format(jsonobj_GOODpRODUCT_value))


    jsonobj_Num_Good_Pro = json.dumps(Num_Good_Pro)
    print("export const jsonobj_Num_Good_Pro = '{}'".format(jsonobj_Num_Good_Pro))
    jsonobj_Num_Bad_Pro = json.dumps(Num_Bad_Pro)
    print("export const jsonobj_Num_Bad_Pro = '{}'".format(jsonobj_Num_Bad_Pro))
    jsonobj_Num_Without_Head = json.dumps(Num_Without_Head)
    print("export const jsonobj_Num_Without_Head = '{}'".format(jsonobj_Num_Without_Head))
    jsonobj_Num_Without_Tail = json.dumps(Num_Without_Tail)
    print("export const jsonobj_Num_Without_Tail = '{}'".format(jsonobj_Num_Without_Tail))
    jsonobj_Total_Num = json.dumps(Total_Num)
    print("export const jsonobj_Total_Num = '{}'".format(jsonobj_Total_Num))





    time.sleep(2)
    # break